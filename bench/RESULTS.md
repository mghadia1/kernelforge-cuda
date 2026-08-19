# Results

**Nothing here is measured yet.** The kernels have not been compiled or run on a
GPU, so every cell below is empty on purpose. No number appears in this file
until it comes out of `bench/results.csv` on a real device, and no claim from
this file reaches the resume before that.

## Setup (to record when the run happens)

| Field | Value |
|---|---|
| GPU | _to fill: e.g. Tesla T4 (sm_75, 15.8 GB), Colab_ |
| CUDA / driver | _to fill_ |
| CPU baseline host | _to fill_ |
| d | 384 (all-MiniLM-L6-v2) |
| k | 5 |
| Repeats | 15 timed, first discarded, 3 warmup |
| Seed | 0 |

## End-to-end latency (median ms, lower is better)

`B = 32`, whole call including host-to-device and device-to-host transfers.

| N | cpu_numpy | v0_naive | v1_shared | v2_warp | v3_topk | cublas | torch_gpu |
|---|---|---|---|---|---|---|---|
| 2,039 | | | | | | | |
| 10,000 | | | | | | | |
| 100,000 | | | | | | | |
| 1,000,000 | | | | | | | |

## Speedup over the CPU baseline (x, median)

| N | v0 | v1 | v2 | v3 | cublas |
|---|---|---|---|---|---|
| 2,039 | | | | | |
| 100,000 | | | | | |

## Where the time goes (v3 vs v2, N = 1,000,000, B = 32)

The point of v3 is that v0-v2 copy a `B x N` score matrix back over PCIe — 128 MB
at this size — to extract 160 numbers. Split the call into transfer, kernel, and
host top-k to show that.

| Stage | v2_warp | v3_topk |
|---|---|---|
| host→device | | |
| kernel | | |
| device→host | | |
| host top-k | | 0 (on GPU) |

## Nsight Compute (v3, N = 100,000, B = 32)

`make profile` runs `ncu --set basic`. Record:

| Metric | Value |
|---|---|
| Achieved occupancy | |
| DRAM throughput (% of peak) | |
| Compute (SM) throughput (%) | |
| Memory-bound or compute-bound | |

Roofline check for a T4 (320 GB/s bandwidth, 8.1 TFLOP/s fp32): the kernel reads
`N x d` floats and does `2 x N x d` FLOPs per query, so arithmetic intensity is
about 0.5 FLOP/byte — far below the T4's ridge point near 25 FLOP/byte. This
problem is memory-bound, and the ceiling is bandwidth, not math. If the measured
DRAM throughput lands near peak, the kernel is done; the remaining gap to cuBLAS
would then be batching and scheduling, not arithmetic.

_State that as a prediction until the profile confirms or refutes it._

## Interpretation (to write after the run)

Answer these, with numbers:

1. What did coalescing buy (v0 → v1), and what did dropping the shared-memory
   round trip buy on top of it (v1 → v2)?
2. At which `N` does moving top-k onto the GPU start to matter, and why is it
   invisible below that?
3. How far behind cuBLAS is v3, and what is cuBLAS doing that v3 is not?
   ("It is hand-tuned" is not an answer.)
4. Where does `torch` sit relative to its own cuBLAS call, and what is the
   overhead in between?
5. What is the failure mode — which `N`, `B`, or `k` makes the current design
   the wrong one?

## Limits

One GPU model, one seed, one `d`, fp32, `k <= 8`, single GPU. Synthetic vectors
above `N = 2039`. Latency only — no throughput-under-load or multi-stream test.
