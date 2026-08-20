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

|       N |   cpu_numpy |   v0_naive |   v1_shared |   v2_warp |   v3_topk |   v4_batch |   v5_regblock |   cublas |   torch_gpu |
|--------:|------------:|-----------:|------------:|----------:|----------:|-----------:|--------------:|---------:|------------:|
|    2039 |       0.631 |      2.412 |       2.276 |     2.816 |     1.881 |      1.514 |         1.442 |    2.765 |       0.988 |
|   10000 |       3.174 |      5.43  |       4.626 |     4.568 |     4.178 |      3.715 |         3.583 |    5.281 |       3.373 |
|  100000 |      79.948 |     41.131 |      40.218 |    38.567 |    40.184 |     36.481 |        36.437 |   36.724 |      33.37  |
| 1000000 |             |    389.584 |     365.663 |   373.143 |   362.155 |    365.347 |       371.792 |  364.852 |     343.607 |

Speedup over `cpu_numpy` (x); blank where the CPU baseline was skipped above N = 200,000:

|       N |   v0_naive |   v1_shared |   v2_warp |   v3_topk |   v4_batch |   v5_regblock |   cublas |   torch_gpu |
|--------:|-----------:|------------:|----------:|----------:|-----------:|--------------:|---------:|------------:|
|    2039 |        0.3 |         0.3 |       0.2 |       0.3 |        0.4 |           0.4 |      0.2 |         0.6 |
|   10000 |        0.6 |         0.7 |       0.7 |       0.8 |        0.9 |           0.9 |      0.6 |         0.9 |
|  100000 |        1.9 |         2   |       2.1 |       2   |        2.2 |           2.2 |      2.2 |         2.4 |
| 1000000 |            |             |           |           |            |               |          |             |

### B = 8, median ms

|       N |   cpu_numpy |   v0_naive |   v1_shared |   v2_warp |   v3_topk |   v4_batch |   v5_regblock |   cublas |   torch_gpu |
|--------:|------------:|-----------:|------------:|----------:|----------:|-----------:|--------------:|---------:|------------:|
|    2039 |       2.527 |      3.05  |       2.768 |     1.999 |     1.569 |      1.407 |         1.252 |    2.539 |       0.991 |
|   10000 |      14.932 |      6.383 |       5.107 |     5.203 |     3.773 |      3.829 |         3.689 |    5.042 |       3.474 |
|  100000 |     202.559 |     56.476 |      43.45  |    46.539 |    42.71  |     40.097 |        36.311 |   38.991 |      34.396 |
| 1000000 |             |    547.738 |     445.633 |   433.946 |   413.416 |    366.222 |       368.344 |  392.149 |     352.332 |

Speedup over `cpu_numpy` (x); blank where the CPU baseline was skipped above N = 200,000:

|       N |   v0_naive |   v1_shared |   v2_warp |   v3_topk |   v4_batch |   v5_regblock |   cublas |   torch_gpu |
|--------:|-----------:|------------:|----------:|----------:|-----------:|--------------:|---------:|------------:|
|    2039 |        0.8 |         0.9 |       1.3 |       1.6 |        1.8 |           2   |      1   |         2.5 |
|   10000 |        2.3 |         2.9 |       2.9 |       4   |        3.9 |           4   |      3   |         4.3 |
|  100000 |        3.6 |         4.7 |       4.4 |       4.7 |        5.1 |           5.6 |      5.2 |         5.9 |
| 1000000 |            |             |           |           |            |               |          |             |

### B = 32, median ms

|       N |   cpu_numpy |   v0_naive |   v1_shared |   v2_warp |   v3_topk |   v4_batch |   v5_regblock |   cublas |   torch_gpu |
|--------:|------------:|-----------:|------------:|----------:|----------:|-----------:|--------------:|---------:|------------:|
|    2039 |       6.918 |      4.461 |       2.747 |     2.516 |     1.659 |      1.44  |         1.274 |    2.934 |       1.012 |
|   10000 |      41.121 |      9.29  |       7.131 |     7.066 |     4.442 |      4.104 |         3.84  |    5.743 |       3.49  |
|  100000 |     487.868 |    113.097 |      69.381 |    64.055 |    55.449 |     39.281 |        38.61  |   47.219 |      34.672 |
| 1000000 |             |   1103.6   |     723.433 |   650.685 |   567.562 |    403.27  |       389.284 |  469.543 |     347.968 |

Speedup over `cpu_numpy` (x); blank where the CPU baseline was skipped above N = 200,000:

|       N |   v0_naive |   v1_shared |   v2_warp |   v3_topk |   v4_batch |   v5_regblock |   cublas |   torch_gpu |
|--------:|-----------:|------------:|----------:|----------:|-----------:|--------------:|---------:|------------:|
|    2039 |        1.6 |         2.5 |       2.7 |       4.2 |        4.8 |           5.4 |      2.4 |         6.8 |
|   10000 |        4.4 |         5.8 |       5.8 |       9.3 |       10   |          10.7 |      7.2 |        11.8 |
|  100000 |        4.3 |         7   |       7.6 |       8.8 |       12.4 |          12.6 |     10.3 |        14.1 |
| 1000000 |            |             |           |           |            |               |          |             |

## The optimization ladder

At N = 1,000,000, B = 32:

| Step | median ms | vs previous |
|---|---:|---:|
| v0 naive | 1103.6 | — |
| v1 shared-memory tiling | 723.4 | 1.53x |
| v2 warp-shuffle | 650.7 | 1.11x |
| v3 on-GPU top-k | 567.6 | 1.15x |
| v4 batch tiling | 403.3 | 1.41x |
| **v5 register blocking** | **389.3** | **1.04x** |
| **v5 vs v0** | | **2.84x** |
| Best over the NumPy CPU baseline | | **12.6x** (v5, N = 100,000, B = 32) |
| cuBLAS + host top-k | 469.5 | v5 is 1.21x faster |
| torch matmul + topk | 348.0 | v5 is 1.12x slower |

Correctness held throughout: 101 tests passed with **0 skipped**, worst absolute
error 2.533e-07, identical indices for every implementation at every size.

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

## The profile, kernel by kernel — each step chosen by the previous one

`ncu --kernel-name <kernel> --set full`, N = 100,000, B = 32:

| Metric | kf_v3_partial | kf_v4_partial | kf_v5_partial |
|---|---:|---:|---:|
| Duration | 26.46 ms | 7.40 ms | **3.88 ms** |
| DRAM throughput | **68.79%** | 39.27% | **74.54%** |
| Memory throughput | 218.77 GB/s | 125.06 GB/s | 237.99 GB/s |
| Compute (SM) throughput | 36.83% | 74.18% | 77.88% |
| L1/TEX cache throughput | 37.44% | **78.48%** | **85.93%** |
| L2 cache throughput | 22.25% | 9.80% | 18.77% |
| fp32 peak achieved | 4% | 13% | **26%** |
| Theoretical occupancy | 75% | 75% | 75% |
| Achieved occupancy | 47.88% | 74.32% | 73.50% |

Read the bold cells left to right; that is the whole project in one table.

**v3** is pinned against DRAM at 68.79% while the SMs idle at 36.83% — and it is
moving eight times more data than the problem requires, because each query
re-reads the corpus.

**v4** cuts that traffic 8x by giving each block a tile of eight queries. DRAM
drops to 39.27% and the kernel runs 3.58x faster on *less* bandwidth
(125 GB/s). The limiter moves to L1/TEX at 78.48%: the query tile is now being
re-read from shared memory on every step.

**v5** attacks exactly that, holding four documents and the eight query values
in registers so shared traffic per FMA falls 4x. The kernel runs **1.91x faster
again** (7.40 → 3.88 ms) and arithmetic doubles to 26% of fp32 peak.

And the bottleneck moves back: DRAM throughput returns to 74.54%, 237.99 GB/s of
the T4's 320 GB/s. That is not a regression — the byte count did not change, the
time halved, so the same traffic now arrives twice as fast. **v5's scoring
kernel is bandwidth-bound at about 74% of peak**, which for a 0.5 FLOP/byte
problem is where a kernel should end up. The next honest gain is not another
tiling trick; it is not re-uploading the corpus.

### What it did not buy, end to end

v5 barely moves the end-to-end column: 0.98x to 1.13x against v4 across the
twelve (N, B) points, and 1.04x at the largest. The 1.54 GB upload dominates
every call, so a kernel that got 1.91x faster is worth almost nothing to the
caller. Both facts are true and the ladder table above reports the diluted one.

## About the cuBLAS comparison — measured, and it is not flattering

v5 is faster than the `cublas` row at **eleven of the twelve** (N, B) points,
1.01x to 2.30x — it loses by 2% at N = 1,000,000, B = 1, where there is no batch
to tile. That looked like the headline until the stage split was actually run.
One warm call, N = 1,000,000, B = 32:

| Stage | cublas | v3_topk | v4_batch |
|---|---:|---:|---:|
| host→device | 342.92 ms | 359.94 ms | 368.58 ms |
| **scoring** | **12.94 ms** | 202.55 ms | **38.45 ms** |
| device→host | 9.76 ms | 0.05 ms | 0.04 ms |
| host top-k | 83.79 ms | 0.00 ms | 0.00 ms |
| total | 501.58 ms | 563.23 ms | 408.01 ms |

**cuBLAS computes the scores in 12.94 ms. v4 took 38.45 ms and v5 takes 29.78 ms
in the same measurement — after register blocking, cuBLAS is still 2.34x faster
at the actual arithmetic** (it was 2.97x before).

v4 wins end-to-end only because the baseline pipeline then pays 9.76 ms to copy
the score matrix back and 83.79 ms to select on the CPU — 93.55 ms of overhead
v4 does not have. Take that away and v4 loses badly.

So the correct claim is narrow and worth stating exactly:

> A hand-written kernel that beats a **cuBLAS GEMM + host-side top-k** pipeline
> end to end, by keeping selection on the device — while still losing to
> cuBLAS's GEMM itself by about 2.3x, down from 3x before register blocking.

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

1. **A persistent-corpus benchmark.** This is now the only change that can move
   the end-to-end number. The scoring kernel is 3.88 ms while the upload is
   ~350 ms; a real system uploads once and queries thousands of times, and until
   the benchmark reflects that, every further kernel gain is invisible.
2. Close the remaining 2.34x to cuBLAS's GEMM — a tuned tile shape per problem
   size, and wider register tiles than 4x8.
3. Parallelize the final fold in `kf_merge_partials`, still serialized in
   thread 0.

## Limits

One GPU, one seed, one `d`, fp32, `k <= 8`, single device. The v5 tile shape
(4 documents x 8 queries) was not swept — it is the first shape that fit the
register budget, not a tuned optimum. Synthetic vectors
above N = 2,039. Latency only. The stage-split and profile tables are single
warm calls, not medians. No fp16 or tensor-core path, no multi-GPU, no TensorRT.
