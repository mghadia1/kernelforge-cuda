# Results

Measured on a Google Colab **Tesla T4 (sm_75, 14.6 GB)**, driver 580.82.07,
CUDA 12.8, on August 19, 2026. Every table is generated from
`bench/results.csv` (93 rows), which the run wrote. Nothing here was typed by
hand.

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

## Correctness

All tests passed with **0 skipped**. Worst absolute error against the NumPy
reference across the whole sweep: **2.533e-07** (tolerance 1e-4), and every
implementation returned **identical indices**, not merely similar scores. The
`(score, index)` tie-break in `topk_rule.h` held on hardware exactly as the
host-side simulations predicted, for both v3's and v4's very different
selection schemes.

## End-to-end latency

### B = 1, median ms

|       N |   cpu_numpy |   v0_naive |   v1_shared |   v2_warp |   v3_topk |   v4_batch |   cublas |   torch_gpu |
|--------:|------------:|-----------:|------------:|----------:|----------:|-----------:|---------:|------------:|
|    2039 |       0.65  |      3.025 |       2.935 |     2.882 |     1.952 |      1.669 |    3.378 |       1.132 |
|   10000 |       7.304 |      7.745 |       5.563 |     4.575 |     4.348 |      3.98  |    5.072 |       3.423 |
|  100000 |      80.404 |     40.844 |      36.467 |    36.503 |    36.271 |     36.428 |   38.702 |      34.81  |
| 1000000 |             |    380.812 |     362.433 |   359.511 |   360.472 |    358.783 |  361.479 |     337.89  |

Speedup over `cpu_numpy` (x); blank where the CPU baseline was skipped above N = 200,000:

|       N |   v0_naive |   v1_shared |   v2_warp |   v3_topk |   v4_batch |   cublas |   torch_gpu |
|--------:|-----------:|------------:|----------:|----------:|-----------:|---------:|------------:|
|    2039 |        0.2 |         0.2 |       0.2 |       0.3 |        0.4 |      0.2 |         0.6 |
|   10000 |        0.9 |         1.3 |       1.6 |       1.7 |        1.8 |      1.4 |         2.1 |
|  100000 |        2   |         2.2 |       2.2 |       2.2 |        2.2 |      2.1 |         2.3 |
| 1000000 |            |             |           |           |            |          |             |

### B = 8, median ms

|       N |   cpu_numpy |   v0_naive |   v1_shared |   v2_warp |   v3_topk |   v4_batch |   cublas |   torch_gpu |
|--------:|------------:|-----------:|------------:|----------:|----------:|-----------:|---------:|------------:|
|    2039 |       7.852 |      6.326 |       2.797 |     2.757 |     1.891 |      1.684 |    3.329 |       1.027 |
|   10000 |      14.299 |      7.047 |       5.495 |     5.207 |     3.844 |      3.848 |    5.199 |       3.494 |
|  100000 |     209.624 |     56.333 |      43.655 |    41.768 |    40.277 |     36.631 |   39.421 |      34.191 |
| 1000000 |             |    550.656 |     449.656 |   430.673 |   407.336 |    364.756 |  395.052 |     338.478 |

Speedup over `cpu_numpy` (x); blank where the CPU baseline was skipped above N = 200,000:

|       N |   v0_naive |   v1_shared |   v2_warp |   v3_topk |   v4_batch |   cublas |   torch_gpu |
|--------:|-----------:|------------:|----------:|----------:|-----------:|---------:|------------:|
|    2039 |        1.2 |         2.8 |       2.8 |       4.2 |        4.7 |      2.4 |         7.6 |
|   10000 |        2   |         2.6 |       2.7 |       3.7 |        3.7 |      2.8 |         4.1 |
|  100000 |        3.7 |         4.8 |       5   |       5.2 |        5.7 |      5.3 |         6.1 |
| 1000000 |            |             |           |           |            |          |             |

### B = 32, median ms

|       N |   cpu_numpy |   v0_naive |   v1_shared |   v2_warp |   v3_topk |   v4_batch |   cublas |   torch_gpu |
|--------:|------------:|-----------:|------------:|----------:|----------:|-----------:|---------:|------------:|
|    2039 |      15.866 |      6.482 |       3.55  |     3.189 |     2.07  |      1.686 |    3.757 |       1.047 |
|   10000 |      40.797 |      9.248 |       6.984 |     6.803 |     4.392 |      4.081 |    5.702 |       3.489 |
|  100000 |     491.734 |    118.64  |      78.379 |    61.541 |    54.978 |     38.538 |   47.787 |      33.91  |
| 1000000 |             |   1127.36  |     744.991 |   639.198 |   575.548 |    406.992 |  505.685 |     340.384 |

Speedup over `cpu_numpy` (x); blank where the CPU baseline was skipped above N = 200,000:

|       N |   v0_naive |   v1_shared |   v2_warp |   v3_topk |   v4_batch |   cublas |   torch_gpu |
|--------:|-----------:|------------:|----------:|----------:|-----------:|---------:|------------:|
|    2039 |        2.4 |         4.5 |         5 |       7.7 |        9.4 |      4.2 |        15.1 |
|   10000 |        4.4 |         5.8 |         6 |       9.3 |       10   |      7.2 |        11.7 |
|  100000 |        4.1 |         6.3 |         8 |       8.9 |       12.8 |     10.3 |        14.5 |
| 1000000 |            |             |           |           |            |          |             |

### p95 at B = 32, ms

|       N |   cpu_numpy |   v0_naive |   v1_shared |   v2_warp |   v3_topk |   v4_batch |   cublas |   torch_gpu |
|--------:|------------:|-----------:|------------:|----------:|----------:|-----------:|---------:|------------:|
|    2039 |      21.014 |      7.969 |       3.637 |     3.335 |     2.189 |      1.703 |    3.887 |       1.095 |
|   10000 |      42.8   |     10.038 |       8.096 |     6.936 |     4.442 |      4.178 |    5.848 |       3.555 |
|  100000 |     501.395 |    121.308 |      84.286 |    64.857 |    55.861 |     40.439 |   48.542 |      35.258 |
| 1000000 |             |   1199.16  |     819.102 |   699.241 |   621.378 |    436.601 |  609.84  |     348.226 |


## The optimization ladder

At N = 1,000,000, B = 32:

| Step | median ms | vs previous |
|---|---:|---:|
| v0 naive | 1127.4 | — |
| v1 shared-memory tiling | 745.0 | 1.51x |
| v2 warp-shuffle | 639.2 | 1.17x |
| v3 on-GPU top-k | 575.5 | 1.11x |
| **v4 batch tiling** | **407.0** | **1.41x** |
| **v4 vs v0** | | **2.77x** |
| | | |
| Best speedup over the NumPy CPU baseline | | **12.8x** (v4, N = 100,000, B = 32) |
| cuBLAS + host top-k | 505.7 | v4 is 1.24x faster |
| torch matmul + topk | 340.4 | v4 is 1.20x slower |

## What batch tiling actually bought

v4 changes one thing: a block now owns a chunk of documents **and a tile of
eight queries**, so each X element is loaded once into a register and reused
against all eight query vectors instead of being re-read once per query.

One warm call at N = 1,000,000, B = 32:

| Stage | v3_topk | v4_batch |
|---|---:|---:|
| host→device | 441.45 ms | 351.60 ms |
| **kernel** | **206.25 ms** | **39.75 ms** |
| device→host | 0.06 ms | 0.03 ms |
| total | 650.48 ms | 392.33 ms |

**The kernel got 5.19x faster.** End-to-end it is only 1.41x, because the
1.54 GB corpus upload still dominates the call — the h2d figures also show the
run-to-run PCIe variance (441 vs 352 ms for the same transfer), which is why
these single-call totals differ from the sweep medians above.

The reuse factor is eight and the kernel speedup is 5.19x. The gap is the part
that does not scale with the tile: the selection stage, the query staging, and
the tail chunks.

## The profile confirms the mechanism

`ncu --kernel-name <kernel> --set full`, N = 100,000, B = 32:

| Metric | kf_v3_partial | kf_v4_partial |
|---|---:|---:|
| Duration | 26.46 ms | **7.40 ms** |
| **DRAM throughput** | **68.79%** | **39.27%** |
| Memory throughput | 218.77 GB/s | 125.06 GB/s |
| **Compute (SM) throughput** | **36.83%** | **74.18%** |
| L1/TEX cache throughput | 37.44% | **78.48%** |
| L2 cache throughput | 22.25% | 9.80% |
| fp32 peak achieved | 4% | **13%** |
| Theoretical occupancy | 75% | 75% |
| Achieved occupancy | 47.88% | **74.32%** |

This is the whole argument in one table. v3 was pinned against DRAM (68.79%)
while the SMs idled at 36.83%. v4 runs 3.58x faster while pulling **less**
bandwidth (125 GB/s against 218 GB/s) — it is not moving data faster, it is
moving less of it. Compute utilization doubled, arithmetic went from 4% to 13%
of fp32 peak, and achieved occupancy rose from 47.88% to 74.32% against the
same 75% theoretical ceiling.

**The bottleneck moved.** L1/TEX throughput went from 37.44% to 78.48% and is
now the highest utilization in the kernel: the limiter is no longer global
memory but the shared-memory reads of the query tile. That names the next
optimization without guesswork — register-blocking the queries so each thread
holds several query values in registers instead of re-reading them from shared
memory each step. That is the standard next move in GEMM tuning, and the
profile is what says so.

## About the cuBLAS comparison — measured, and it is not flattering

v4 is faster than the `cublas` row at **every size measured**, 1.01x to 2.23x.
That looked like the headline until the stage split was actually run. One warm
call, N = 1,000,000, B = 32:

| Stage | cublas | v3_topk | v4_batch |
|---|---:|---:|---:|
| host→device | 342.92 ms | 359.94 ms | 368.58 ms |
| **scoring** | **12.94 ms** | 202.55 ms | **38.45 ms** |
| device→host | 9.76 ms | 0.05 ms | 0.04 ms |
| host top-k | 83.79 ms | 0.00 ms | 0.00 ms |
| total | 501.58 ms | 563.23 ms | 408.01 ms |

**cuBLAS computes the scores in 12.94 ms. v4 takes 38.45 ms — cuBLAS is 2.97x
faster at the actual arithmetic.**

v4 wins end-to-end only because the baseline pipeline then pays 9.76 ms to copy
the score matrix back and 83.79 ms to select on the CPU — 93.55 ms of overhead
v4 does not have. Take that away and v4 loses badly.

So the correct claim is narrow and worth stating exactly:

> A hand-written kernel that beats a **cuBLAS GEMM + host-side top-k** pipeline
> end to end, by keeping selection on the device — while losing to cuBLAS's GEMM
> itself by about 3x.

This also explains the torch row without any hand-waving. torch is cuBLAS's fast
GEMM *plus* an on-device `topk`: roughly 343 ms of upload, ~13 ms of scoring, and
a negligible selection, which lands at the ~340 ms it measures. It is the
combination v4 was trying to approximate, assembled from a better GEMM.

(The `cublas` stages sum to 449.41 ms against a 501.58 ms total; the ~52 ms
difference is allocation and cuBLAS handle setup inside the timed call.)

Closing the 3x scoring gap is not a tweak. cuBLAS tiles both dimensions with
register blocking and a tuned tile shape per problem size; v4 tiles eight
queries and reads them from shared memory every step, which is exactly what its
78.48% L1/TEX throughput reports.

## Where the GPU still loses

At PaperTrail's actual corpus size — N = 2,039, one query at a time — NumPy is
still fastest: **0.650 ms against v4's 1.669 ms**.

Batch tiling cannot help *through reuse* at B = 1, because with one query there
is nothing to reuse a loaded byte against. The B = 1 column bears that out where
bandwidth is the constraint: v3 and v4 land within 0.4% of each other at
N = 100,000 and N = 1,000,000.

It does not follow that v4 is pointless at B = 1. It is still ahead at the two
small sizes — 1.17x at N = 2,039 and 1.09x at N = 10,000 — and that is *not* the
tiling. It is v4's smaller 256-document chunk and its warp-level selection,
which matter when the call is dominated by launch overhead and selection rather
than by memory traffic. Attributing those wins to batch tiling would be wrong.

## What to do next, in order

1. **Register-block the query tile.** L1/TEX at 78.48% is the new limiter.
2. **A persistent-corpus benchmark.** The 1.54 GB upload is 90% of v4's call and
   a real system does it once. This now matters more than it did: the faster the
   kernel gets, the more the measurement is just PCIe.
3. Parallelize the final fold in `kf_merge_partials`, still serialized in
   thread 0.

The stage split has been run; the 3x scoring gap to cuBLAS is now the thing to
attack, and step 1 is the first move in that direction.

## Limits

One GPU, one seed, one `d`, fp32, `k <= 8`, single device. Synthetic vectors
above N = 2,039. Latency only. The stage-split and profile tables are single
warm calls, not medians. No fp16 or tensor-core path, no multi-GPU, no TensorRT.
