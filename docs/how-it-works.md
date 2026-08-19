# How KernelForge works

## The hot path it replaces

PaperTrail answers a question by embedding it with `all-MiniLM-L6-v2` (384
dimensions), scoring that vector against every stored chunk, and keeping the
best five. Embeddings are L2-normalized, so cosine similarity is a plain dot
product and the whole retrieval step is:

```
scores = q @ X.T      # (B, 384) x (384, N)  ->  (B, N)
top5   = argsort(-scores)[:, :5]
```

That is the entire problem KernelForge implements in CUDA. It is worth doing
because it is small enough to write four honest versions of, and because the
arithmetic intensity is low enough (about 0.5 FLOP per byte) that it is a clean
lesson in memory-bound optimization rather than a lesson in FLOPs.

## One ABI, five implementations

Everything sits behind the C interface in `src/kernelforge.h`:

```c
int kf_v0_naive(const float *q, const float *X, int B, int N, int d, int k,
                float *out_vals, int *out_idx, KfTiming *timing);
```

`v1_shared`, `v2_warp`, `v3_topk`, and `cublas` have identical signatures. That
uniformity is what makes the benchmark trustworthy: `bench/run.py` reaches each
one through the same ctypes path, hands it the same arrays, and measures the
same thing — end-to-end wall time for the whole call, transfers included.

Every wrapper also fills a `KfTiming` splitting the call into host-to-device,
kernel, device-to-host, and host top-k. That split is the evidence for v3.

## The four versions, and what each one is arguing

### v0 — naive
One thread per (query, document). Each thread walks a whole 384-float row alone.
Neighbouring threads work on neighbouring *documents*, so at every step of the
inner loop they touch addresses 384 floats apart: one warp load becomes 32
separate cache lines. This is deliberately the worst reasonable arrangement. It
exists to be correct and to be the number the others beat.

### v1 — shared-memory tiling
A block stages a tile of `X` (128 documents x 32 dimensions) into shared memory
using *coalesced* reads — consecutive threads read consecutive dimensions inside
one row — then each thread consumes its own document's row out of shared memory.

The subtle part is the row stride. The tile is stored with a stride of 33 floats,
not 32. During the compute phase thread `t` reads `sX[t][i]` for a fixed `i`; at
stride 32 every one of those addresses lands in the same shared-memory bank, so
the access serializes 32 ways and the tiling buys back roughly nothing. Padding
by one float rotates each row into a different bank. It is a one-character change
that decides whether v1 is an improvement at all.

### v2 — warp per document
v1 fixed coalescing by paying for a shared-memory round trip: every element of
`X` is written to shared and read straight back, with a `__syncthreads` per tile.

In v2 an entire warp owns one document. Lane `l` reads `X[n][l]`, `X[n][l+32]`,
... directly from global memory, which is already perfectly coalesced — 32 lanes
covering 128 contiguous bytes — so `X` never enters shared memory. The 32 partial
sums then collapse through `__shfl_down_sync`, a register-to-register exchange
inside the warp that needs no shared memory and no barrier at all. Only `q`,
which every warp in the block re-reads, is staged in shared memory.

### v3 — top-k on the GPU
v0 through v2 all copy a `B x N` score matrix back to the host and select there.
At `N = 1,000,000` and `B = 32` that is 128 MB crossing PCIe to extract 160
numbers, plus a serial host scan over 32M floats.

v3 keeps selection on the device in two stages:

1. **Partial** — each block scores a chunk of 1024 documents (v2's warp-per-doc
   scheme), holds those scores in shared memory, and reduces them to its own
   top-k: 256 threads each keep a register list over a strided slice, then one
   thread merges the 256 lists. Output is `k` candidates per block.
2. **Merge** — one block per query folds every block's candidates into the final
   `k` the same way.

Only `B x k` values ever come back.

## The one comparison rule

A two-stage selection has a hazard the earlier versions do not: its answer could
depend on the order blocks happen to finish in. Two documents with identical
scores would be resolved by whichever candidate arrived first, and the result
would wobble between runs.

`src/topk_rule.h` removes that. It defines a single comparison —

> `(s, i)` beats `(v, j)` when `s > v`, or when `s == v` and `i < j`

— compiled `__host__ __device__` and used by the host fallback in `common.cuh`,
by both stages of v3, and by the CPU simulation. Ties always go to the lower
document index, which is exactly what `numpy.argsort(..., kind="stable")` gives
`reference.py`. Because both sides agree, the tests can assert on returned
*indices*, not merely on scores — a much sharper check.

## What is verified, and where

| Layer | Checked by | Needs a GPU? |
|---|---|---|
| The reference itself | `test_correctness.py` vs a plain Python loop | no |
| The comparison rule | `test_selection.py` against `topk_rule.h` directly | no |
| v3's two-stage decomposition | `test_selection.py` via `selection_sim.cpp` | no |
| The kernels compile | CI, in a CUDA devel container | no |
| The kernels are *correct* | `test_correctness.py`, GPU path | **yes** |
| Any performance claim | `bench/run.py` on a T4 | **yes** |

`src/selection_sim.cpp` deserves a note. It runs v3's exact decomposition — same
chunk size, same 256-thread strides, same per-block merge, same final merge —
serially in plain C++, sharing the constants in `v3_config.h` and the rule in
`topk_rule.h` so it cannot drift from the kernel. That lets the awkward cases be
attacked on any laptop: `N` on and either side of a chunk boundary, every
supported `k`, all-identical scores, scores quantized to 8 levels so ties cross
block boundaries, winners hidden in the short tail chunk, and winners planted one
per chunk so the merge must read every partial list.

It narrows where a bug can hide. It does not prove the kernel correct: shared
memory hazards, the missing-`__syncthreads` class of bug, launch configuration,
occupancy, and the dot product arithmetic are all invisible to it. Only the T4
run closes those.

## Benchmark discipline

Warm up, then discard the first timed run — it pays for allocation, context
setup, and the cuBLAS handle, and keeping it would flatter whichever
implementation happened to run second. At least 15 repeats, report median and
p95 rather than the minimum, and record the device in every row so two machines
never end up in one table. Each implementation is checked against the reference
at each size *before* it is timed, so a fast wrong answer can never be published
as a speedup.

## Known limits

One GPU model, one seed, one embedding dimension, fp32 only, `k <= 8` (the
device-side lists reserve shared memory for that bound; `kf_v3_topk` returns -1
above it rather than writing past the allocation). Synthetic vectors above
`N = 2039`. No fp16 or tensor-core path, no multi-GPU, no TensorRT. Latency only
— no throughput-under-load or multi-stream measurement.
