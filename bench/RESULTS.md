# Results

Measured on a Google Colab **Tesla T4 (sm_75, 14.6 GB)**, driver 580.82.07,
CUDA 12.8 (`nvcc` release 12.8, V12.8.93), on August 19, 2026. Every table below
is generated from `bench/results.csv`, which the run wrote; nothing here was
typed by hand.

## Setup

| Field | Value |
|---|---|
| GPU | Tesla T4 (sm_75, 14.6 GB), Google Colab |
| Toolkit / driver | CUDA 12.8 / 580.82.07 |
| d | 384 (all-MiniLM-L6-v2) |
| k | 5 |
| Repeats | 15 timed, first discarded, 3 warmup |
| Seed | 0 |
| Timing | end-to-end wall time per call, transfers included |
| Rows | 81 |

## Correctness

All 57 tests passed with **0 skipped** — the first time any of these kernels had
executed. Worst absolute error against the NumPy reference across the entire
sweep was **2.533e-07** (tolerance 1e-4), and every implementation returned
**identical indices**, not merely similar scores. The `(score, index)` tie-break
rule in `topk_rule.h` held on hardware exactly as the host-side simulation
predicted.

## End-to-end latency

### B = 1, median ms

|       N |   cpu_numpy |   v0_naive |   v1_shared |   v2_warp |   v3_topk |   cublas |   torch_gpu |
|--------:|------------:|-----------:|------------:|----------:|----------:|---------:|------------:|
|    2039 |       0.601 |      2.377 |       2.312 |     2.212 |     2.03  |    2.88  |       1.003 |
|   10000 |       3.026 |      5.167 |       4.68  |     4.997 |     4.261 |    4.967 |       3.468 |
|  100000 |      81.565 |     42.098 |      41.67  |    37.21  |    36.592 |   36.954 |      33.328 |
| 1000000 |             |    386.259 |     362.691 |   371.631 |   361.519 |  368.469 |     336.569 |

Speedup over `cpu_numpy` (x); blank where the CPU baseline was skipped above N = 200,000:

|       N |   v0_naive |   v1_shared |   v2_warp |   v3_topk |   cublas |   torch_gpu |
|--------:|-----------:|------------:|----------:|----------:|---------:|------------:|
|    2039 |        0.3 |         0.3 |       0.3 |       0.3 |      0.2 |         0.6 |
|   10000 |        0.6 |         0.6 |       0.6 |       0.7 |      0.6 |         0.9 |
|  100000 |        1.9 |         2   |       2.2 |       2.2 |      2.2 |         2.4 |
| 1000000 |            |             |           |           |          |             |

### B = 8, median ms

|       N |   cpu_numpy |   v0_naive |   v1_shared |   v2_warp |   v3_topk |   cublas |   torch_gpu |
|--------:|------------:|-----------:|------------:|----------:|----------:|---------:|------------:|
|    2039 |       2.672 |      3.33  |       2.831 |     2.736 |     1.788 |    2.528 |       0.989 |
|   10000 |      14.119 |      7.188 |       5.09  |     5.023 |     4.216 |    5.068 |       3.416 |
|  100000 |     209.189 |     56.642 |      43.638 |    45.12  |    41.226 |   43.779 |      33.902 |
| 1000000 |             |    547.232 |     444.445 |   421.646 |   408.485 |  401.193 |     338.312 |

Speedup over `cpu_numpy` (x); blank where the CPU baseline was skipped above N = 200,000:

|       N |   v0_naive |   v1_shared |   v2_warp |   v3_topk |   cublas |   torch_gpu |
|--------:|-----------:|------------:|----------:|----------:|---------:|------------:|
|    2039 |        0.8 |         0.9 |       1   |       1.5 |      1.1 |         2.7 |
|   10000 |        2   |         2.8 |       2.8 |       3.3 |      2.8 |         4.1 |
|  100000 |        3.7 |         4.8 |       4.6 |       5.1 |      4.8 |         6.2 |
| 1000000 |            |             |           |           |          |             |

### B = 32, median ms

|       N |   cpu_numpy |   v0_naive |   v1_shared |   v2_warp |   v3_topk |   cublas |   torch_gpu |
|--------:|------------:|-----------:|------------:|----------:|----------:|---------:|------------:|
|    2039 |       7.094 |      5.288 |       2.762 |     2.416 |     1.887 |    2.692 |       1.008 |
|   10000 |      39.901 |     10.146 |       7.039 |     6.932 |     4.522 |    5.476 |       3.444 |
|  100000 |     501.381 |    112.51  |      68.01  |    63.386 |    55.593 |   47.851 |      34.522 |
| 1000000 |             |   1110     |     735.101 |   637.119 |   563.64  |  461.71  |     343.675 |

Speedup over `cpu_numpy` (x); blank where the CPU baseline was skipped above N = 200,000:

|       N |   v0_naive |   v1_shared |   v2_warp |   v3_topk |   cublas |   torch_gpu |
|--------:|-----------:|------------:|----------:|----------:|---------:|------------:|
|    2039 |        1.3 |         2.6 |       2.9 |       3.8 |      2.6 |         7   |
|   10000 |        3.9 |         5.7 |       5.8 |       8.8 |      7.3 |        11.6 |
|  100000 |        4.5 |         7.4 |       7.9 |       9   |     10.5 |        14.5 |
| 1000000 |            |             |           |           |          |             |

### p95 at B = 32, ms

|       N |   cpu_numpy |   v0_naive |   v1_shared |   v2_warp |   v3_topk |   cublas |   torch_gpu |
|--------:|------------:|-----------:|------------:|----------:|----------:|---------:|------------:|
|    2039 |       7.463 |      5.337 |       2.87  |     2.485 |     1.915 |    2.783 |       1.044 |
|   10000 |      41.502 |     11.077 |       7.225 |     7.297 |     5.082 |    5.71  |       3.548 |
|  100000 |     677.438 |    113.536 |      72.223 |    65.751 |    56.288 |   50.329 |      35.599 |
| 1000000 |             |   1200.85  |     803.82  |   641.242 |   567.689 |  562.768 |     380.382 |


## What the optimization ladder was worth

At the largest size (N = 1,000,000, B = 32) each step paid:

| Step | median ms | vs previous |
|---|---:|---:|
| v0 naive | 1110.0 | — |
| v1 shared-memory tiling | 735.1 | 1.51x |
| v2 warp-shuffle | 637.1 | 1.15x |
| v3 on-GPU top-k | 563.6 | 1.13x |
| **v3 vs v0** | | **1.97x** |
| cuBLAS | 461.7 | v3 is 1.22x slower |
| torch matmul + topk | 343.7 | v3 is 1.64x slower |

**The library wins, and by a knowable amount.** v3 is 22% behind cuBLAS and 64%
behind PyTorch. PyTorch beating its own cuBLAS backend is the interesting part:
`torch.topk` runs on the device and `torch` reuses cached device allocations,
where the `cublas` row here pays a fresh `cudaMalloc` and a host top-k on every
call.

Best speedup over the NumPy CPU baseline: **9.0x** (v3, N = 100,000, B = 32).
That is far short of the order-of-magnitude the project spec anticipated, and
the next section explains why.

## Where the time actually goes — the prediction was wrong

One warm measured call at N = 1,000,000, B = 32:

| Stage | v2_warp | v3_topk |
|---|---:|---:|
| host→device | 325.26 ms | 364.33 ms |
| kernel | 179.57 ms | 207.29 ms |
| device→host | 9.95 ms | 0.05 ms |
| host top-k | 57.88 ms | 0.00 ms |
| **total** | **627.70 ms** | **572.65 ms** |

`RESULTS.md` predicted before the run that v3's win would come from deleting a
128 MB device-to-host copy. **That prediction was wrong.** The copy costs 9.95 ms
— under 2% of the call. v3's real saving is the 57.88 ms of *host-side
selection* it removes, and its own kernel is 27.7 ms **slower** than v2's because
it now does the selection too. Net 55 ms, which matches the sweep.

The dominant cost is the one nobody optimized: **the host→device upload, 325 ms,
52% of the call.** At N = 1,000,000 and d = 384 the corpus is 1.54 GB, and this
benchmark re-uploads it on *every* call. A real retrieval system uploads a corpus
once and queries it thousands of times.

That makes the end-to-end column above a measurement of PCIe bandwidth as much
as of the kernels, and it compresses the differences between every GPU row. The
honest reading: the ladder ranking is real, the magnitudes are understated, and
a persistent-corpus benchmark is the single most valuable thing this project
still lacks.

## Where the GPU loses outright

At PaperTrail's actual corpus size — N = 2,039 chunks — with one query at a
time, **NumPy wins**: 0.601 ms against v3's 2.030 ms. Every GPU path loses
below roughly N = 100,000 at B = 1, because a kernel launch plus two transfers
costs more than the arithmetic saved.

The GPU becomes worth it in two situations, both visible in the tables:
batching (at N = 2,039, B = 32, v3 is 3.8x faster than NumPy) and scale
(at N = 100,000, B = 32, 9.0x).

So the defensible claim is not "I made retrieval faster." It is: *on the corpus
PaperTrail actually has, this kernel is slower than NumPy unless queries are
batched; it pays off from about 10^5 documents.*

## Nsight Compute

Two captures, and the first one misled. `ncu --set basic` profiles whichever
kernel it reaches first and reported 3.80% achieved occupancy against 100%
theoretical — numbers that match neither of v3's kernels' shared-memory budget.
The targeted capture below reports 75% theoretical for `kf_v3_partial`, so the
basic run had profiled the *other* kernel, `kf_v3_merge`, which launches only
B = 32 blocks across the T4's 40 SMs. Its load-imbalance warning ("minimum
instance value is 100.00% below the average") is exactly that: idle SMs.

Lesson recorded: an `ncu` number is meaningless without the kernel name attached
to it.

### `kf_v3_partial` — the scoring kernel

`ncu --kernel-name kf_v3_partial --set full`, grid (98, 32, 1), block (256, 1, 1),
31 passes:

| Metric | Value |
|---|---:|
| **DRAM throughput** | **68.79% of peak** |
| Memory throughput | 218.77 GB/s |
| **Compute (SM) throughput** | **36.83%** |
| fp32 peak achieved | 4% |
| L1/TEX cache throughput | 37.44% |
| L2 cache throughput | 22.25% |
| Theoretical occupancy | 75% |
| Achieved occupancy | 47.88% |
| Executed IPC | 0.78 |
| SM busy | 19.90% |
| Duration | 26.46 ms |

**The prediction holds.** The scoring kernel is memory-bound: DRAM throughput is
roughly double the SM throughput, arithmetic reaches 4% of the T4's fp32 peak,
and Nsight's own verdict is "Memory is more heavily utilized than Compute." At
218.77 GB/s against the T4's 320 GB/s the kernel is running at about 69% of the
bandwidth ceiling, which is where a 0.5 FLOP/byte problem belongs.

### Why that also explains the cuBLAS gap

Being near the bandwidth ceiling means the only way left to go faster is to
**move fewer bytes** — and this kernel moves far more than it needs to. One
block owns one (query, chunk) pair, so every query re-reads the whole corpus:
at N = 100,000 that is 153.6 MB of X read **32 times**, about 4.9 GB, to answer
32 queries.

cuBLAS does not win by being faster per byte. It wins by tiling over *both*
dimensions, holding a tile of X in shared memory and reusing it across many
queries, so it moves roughly B times less data. That is a specific, checkable
explanation for the 1.22x gap, and it is the next optimization: tile the batch
dimension so each X tile is loaded once and scored against several queries.

75% theoretical occupancy is the shared-memory reservation (the query vector
plus KF_MAX_K-wide candidate lists per thread); 47.88% achieved against it is
the tail effect of chunks that finish early. Neither is the binding constraint
while DRAM sits at 69%.

## What to do next, in order

1. **Tile the batch dimension.** The profile says the kernel is bandwidth-bound
   while re-reading X once per query. Loading each X tile once and scoring it
   against several queries is the change with a measured argument behind it,
   and it is what cuBLAS is doing differently.
2. A persistent-corpus benchmark: upload X once, time only the query calls.
   Both the realistic case and the one where kernel differences stop hiding
   behind 325 ms of PCIe.
3. Parallelize the final fold in `kf_v3_merge` instead of serializing it in
   thread 0 — worth little in total time, but it is a real design weakness and
   the 32-block launch leaves most of the GPU idle.

## Limits

One GPU, one seed, one `d`, fp32, `k <= 8`, single device. Synthetic vectors
above N = 2,039. Latency only — no throughput-under-load, no multi-stream. The
stage-split table is one warm call, not a median over repeats. No fp16 or
tensor-core path, no multi-GPU, no TensorRT.
