# KernelForge — a custom CUDA retrieval kernel, benchmarked honestly

PaperTrail's retrieval step is a batched cosine similarity over 384-dimensional
`all-MiniLM-L6-v2` embeddings followed by a top-k. KernelForge rewrites that hot
path as a hand-written CUDA kernel, optimizes it through four versions, and
measures it against a CPU baseline, PyTorch, and cuBLAS.

The goal is not to beat NVIDIA's libraries. It is to show GPU computing,
parallel programming, and performance analysis with numbers that survive
questions — including the ones about where the library wins.

## Status: `resume_eligible: no` — one gate left

**Ran on a Colab Tesla T4.** 101 tests pass with **0 skipped**, worst absolute
error 2.533e-07, every implementation returning indices identical to the NumPy
reference. `bench/results.csv` holds 105 measured rows;
[`bench/RESULTS.md`](bench/RESULTS.md) interprets them.

N = 1,000,000, B = 32, end-to-end median:

| v0 | v1 | v2 | v3 | v4 | **v5** | cuBLAS+host top-k | torch |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1103.6 | 723.4 | 650.7 | 567.6 | 403.3 | **389.3 ms** | 469.5 | 348.0 |

2.84x over the naive kernel, 12.6x over the NumPy CPU baseline at its best point.

**Every step was chosen by the previous step's profile**, and the scoring kernel
tells that story better than the wall-clock does (`ncu`, N = 100,000, B = 32):

| | kf_v3_partial | kf_v4_partial | kf_v5_partial |
|---|---:|---:|---:|
| Duration | 26.46 ms | 7.40 ms | **3.88 ms** |
| DRAM throughput | **68.79%** | 39.27% | **74.54%** |
| Compute (SM) | 36.83% | 74.18% | 77.88% |
| L1/TEX | 37.44% | **78.48%** | **85.93%** |
| fp32 peak | 4% | 13% | **26%** |

v3 was DRAM-bound while moving 8x more data than necessary. v4 tiled the batch
so each X byte serves eight queries — 3.58x faster on *less* bandwidth, which
pushed the limiter to shared-memory reads of the query tile (L1/TEX 78.48%). v5
register-blocked exactly that: 1.91x faster again, arithmetic doubled to 26% of
peak, and DRAM back to 74.54% — the same bytes arriving in half the time. The
kernel is now bandwidth-bound at ~74% of the T4's 320 GB/s, which is where a
0.5 FLOP/byte problem belongs.

Honest qualifiers, stated up front:

- **cuBLAS's GEMM is still 2.34x faster than v5's kernel** — 12.94 ms against
  29.78 ms at N = 1M, B = 32 (it was 2.97x before register blocking). v5 wins
  the end-to-end row only because that baseline then spends 9.76 ms copying
  scores back and 83.79 ms selecting on the CPU. The claim is "beats a cuBLAS
  GEMM + **host-side** top-k pipeline", never "beats cuBLAS". PyTorch — the same
  fast GEMM with a device-side `topk` — still wins at 348.0 ms.
- **v5 barely moves the end-to-end number** (1.04x over v4) because a 1.54 GB
  corpus upload dominates every call. A 1.91x kernel gain worth almost nothing
  to the caller is the honest result, and it makes the persistent-corpus
  benchmark the only remaining lever.
- **At PaperTrail's real size the GPU still loses**: N = 2,039 with one query,
  NumPy is fastest.
- Two predictions this repo made before the first run were wrong and are
  corrected in RESULTS.md rather than deleted.

Four of five conditions are met: it builds, it passes with 0 skips, the
benchmark is recorded, and the profiles are captured and interpreted. The fifth
is Mayank's — explaining one design choice and one failure mode unaided.
**Until that closes, CUDA stays off the resume and unchecked on applications.**

## The problem

Given a query matrix `q` (`B x 384`) and a corpus `X` (`N x 384`), all rows L2
normalized so cosine similarity is a plain dot product, return the `k = 5`
highest-scoring documents per query. `N` starts at 2,039 — PaperTrail's real
chunk count — and scales to 1M synthetic rows.

## The four versions

| # | File | Idea | What it should show |
|---|------|------|---------------------|
| 0 | [`src/v0_naive.cu`](src/v0_naive.cu) | One thread per (query, document). No caching, no coalescing. | Correctness, and the number to beat. |
| 1 | [`src/v1_shared.cu`](src/v1_shared.cu) | Stage a padded tile of `X` in shared memory with coalesced loads. | What fixing the access pattern is worth. |
| 2 | [`src/v2_warp.cu`](src/v2_warp.cu) | One warp per document; coalesced global reads reduced with `__shfl_down_sync`. | That the shared-memory round trip in v1 was avoidable. |
| 3 | [`src/v3_topk.cu`](src/v3_topk.cu) | v2 scoring plus per-block partial top-k and a merge kernel. | End-to-end latency once the `B x N` PCIe copy is gone. |
| 4 | [`src/v4_batch.cu`](src/v4_batch.cu) | Tile the batch: each block scores a chunk against 8 queries, so every X element loaded is reused 8 times. Selection becomes a warp-per-query masked max-reduction with no shared scratch. | Whether cutting DRAM traffic 8x moves the roofline. It does. |
| 5 | [`src/v5_regblock.cu`](src/v5_regblock.cu) | Register-block it: a warp holds 4 documents and the 8 query values in registers, so shared-memory traffic per FMA falls 4x. | Whether the L1 pressure v4 created is the real limiter. It is — 1.91x. |
| — | [`src/cublas_ref.cu`](src/cublas_ref.cu) | `cublasSgemm` + host top-k, same ABI and timing points. | The honest gap to a tuned library. |

Every selection in the project — host fallback, both stages of v3, and the CPU
simulation — goes through one comparison in
[`src/topk_rule.h`](src/topk_rule.h): a candidate wins on a higher score, or on
an equal score with a lower document index. Ties never depend on which block
finished first, which is what lets the tests compare *indices* and not just
values.

All five sit behind one C ABI ([`src/kernelforge.h`](src/kernelforge.h)) and are
driven from Python through ctypes ([`src/runner.py`](src/runner.py)), so the
benchmark compares like with like.

## Quickstart (Colab / Kaggle T4)

```bash
git clone https://github.com/mghadia1/kernelforge-cuda && cd kernelforge-cuda
pip install -e '.[dev]'
make                 # builds build/libkernelforge.so for sm_75 (T4)
python -m pytest     # every kernel vs the NumPy reference, max abs err < 1e-4
make bench           # sweeps N and B into bench/results.csv
```

Or open [`colab/KernelForge_T4.ipynb`](colab/KernelForge_T4.ipynb) in Colab, pick a T4
runtime, and run it top to bottom: it builds, gates on correctness, sweeps the
benchmark, splits v2 vs v3 by transfer/kernel/top-k, runs Nsight Compute, and
prints the markdown tables to paste into `bench/RESULTS.md`.

On a machine with a different GPU, pass the arch: `make ARCH=sm_80` (A100),
`make ARCH=sm_89` (L4). Without `nvcc` the build is skipped, the GPU tests skip
with a reason, and only the NumPy reference is exercised.

## Correctness

[`src/reference.py`](src/reference.py) is the ground truth: a dense matmul and a
stable argsort. Every kernel must match it within `1e-4` absolute error *and*
return the same indices — which requires one shared rule, that ties go to the
lower document index. The device-side selection implements it as a
`(score, index)` lexicographic comparison so the answer does not depend on the
order blocks happen to finish in.

The test sizes deliberately include the awkward ones: `N = 2039` (not a multiple
of any tile), `N = 1` (less than one warp of work), and `N = 1024` (exactly one
v3 chunk, so the tail path is empty).

v3 is the only version whose answer depends on how work is split across blocks,
so it gets a second line of defence that needs no GPU:
[`src/selection_sim.cpp`](src/selection_sim.cpp) runs its exact decomposition
serially in plain C++ — same chunk size, same strides, same merge, sharing
[`src/v3_config.h`](src/v3_config.h) and the rule above so it cannot drift from
the kernel. [`tests/test_selection.py`](tests/test_selection.py) then attacks it
with all-identical scores, ties quantized across six chunks, winners buried in
the short tail chunk, and winners planted one per chunk so the merge has to read
every partial list. Build it with `make sim`; the tests compile it on demand.

## Benchmark discipline

Same GPU, same `d`, sweep `N` and `B`, warm up, at least 15 repeats, report
median and p95, and discard the first timed run — it pays for allocation and
cuBLAS handle setup. Every implementation is verified against the reference at
each size before it is timed, so a fast wrong answer can never be reported as a
speedup. Every row records the device it ran on.

Timing is end-to-end wall time for the whole call, transfers included. A
kernel-only number would hide the `B x N` device-to-host copy that v3 exists to
remove, which would make v3 look pointless.

## Limits (stated up front)

One GPU model, one seed, one embedding dimension, `k <= 8`, fp32 only. No
tensor-core or fp16 path, no multi-GPU, no NCCL, no TensorRT. Synthetic vectors
above `N = 2039`: the timing is bandwidth-bound and does not depend on what the
vectors mean, but that also means this measures kernel speed, not retrieval
quality.

## Layout

```
src/         kernelforge.h, common.cuh, topk_rule.h, v3_config.h,
             v0-v3 + cublas_ref, selection_sim.cpp, reference.py, runner.py
docs/        how-it-works.md - the optimization story and what is verified where
bench/       run.py (sweep) and RESULTS.md (interpretation)
colab/       KernelForge_T4.ipynb - the whole run on a Colab T4
tests/       test_correctness.py (kernels vs reference), test_selection.py (v3 logic, no GPU)
Makefile     nvcc build, `make sim`, `make test`, `make bench`, `make profile`
Dockerfile   the CI build/verify environment; compiles but cannot run the kernels
```

See [`PROJECT_SPEC.md`](PROJECT_SPEC.md) for the full plan and
[`bench/RESULTS.md`](bench/RESULTS.md) for results once they exist.
