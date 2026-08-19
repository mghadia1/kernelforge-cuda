# KernelForge — a custom CUDA retrieval kernel, benchmarked honestly

A learning-to-production CUDA project: rewrite the retrieval hot path from **PaperTrail**
(batched cosine similarity + top-k over embeddings) as a hand-written CUDA kernel,
optimize it through several versions, and benchmark it rigorously against a CPU baseline,
PyTorch on GPU, and cuBLAS.

The point is not to beat NVIDIA's libraries (you won't, and saying so is a strength). The
point is to demonstrate **GPU computing, parallel programming, and performance
profiling/analysis** — the exact focus areas NVIDIA's Deep Learning / GPU roles ask for —
with numbers you can defend.

## Honesty boundary (read first)

> **`resume_eligible: no` until it is built, run, and benchmarked on a real GPU.**
> Do not add CUDA to the resume, or check CUDA / GPU-computing boxes on applications,
> until v0–v3 below actually run and produce measured results. When that's true, and you
> can explain one design choice and one failure mode unaided, it becomes a strong,
> defensible addition — same standard as LoRAForge, PaperTrail, and LinkForge.

## Why this project

- **Extends a project you already own.** "I profiled my RAG system, found retrieval was the
  bottleneck, and moved it to a custom GPU kernel" is a real, coherent story — not a toy.
- **Hits NVIDIA's named focus areas:** GPU computing, parallel programming, performance
  modeling / profiling / optimization.
- **Measurable and honest.** You get a clean speedup curve, an Nsight profile, and a
  roofline — which is how you already work.

## The problem

Given a query embedding `q` (dim `d = 384`, from all-MiniLM-L6-v2) and a corpus matrix
`X` of `N` document embeddings (start with `N = 2,039`, then scale to 100k–1M synthetic
rows), compute:

1. **Cosine similarity** of `q` against every row of `X` (embeddings are normalized, so this
   is a dot product).
2. **Top-k** (k = 5) highest-scoring rows.

Batch it over `B` queries. This is exactly PaperTrail's vector-search step.

## Milestones (correctness first, then speed)

| # | Version | What you build | What you measure |
|---|---------|----------------|------------------|
| 0 | **Correct baseline** | Naive CUDA kernel: one thread per (query, doc) dot product; copy scores back, top-k on CPU. Validate against a NumPy reference (max abs error < 1e-4). | Correctness only. |
| 1 | **Shared-memory tiling** | Tile `X` and `q` into shared memory; each block handles a tile of documents. | Latency vs v0. |
| 2 | **Warp-shuffle reduction** | Reduce the per-row dot product with `__shfl_down_sync` instead of shared-memory-only reduction; coalesce global loads. | Latency vs v1. |
| 3 | **On-GPU top-k** | Move top-k onto the GPU (per-block partial top-k, then merge) so you don't ship all `N` scores back. | End-to-end latency. |
| 4 | **Benchmark & profile** | Compare all versions to: (a) CPU NumPy, (b) `torch` matmul+topk on GPU, (c) cuBLAS GEMM + topk. | Speedups + roofline. |

## Benchmark plan (this is the resume-worthy part)

- **Fix the setup:** same GPU (Colab/Kaggle **T4**), same `d`, sweep `N` and `B`, warm up,
  run ≥ 15 repeats, report median and p95, and **exclude the first (JIT/allocation) run** —
  the same discipline you used in SensorGuard/LinkForge benchmarks.
- **Report honestly:**
  - Your best kernel's speedup over the CPU baseline (this will be large — memory-bandwidth
    bound problem).
  - Where you **lose to cuBLAS/torch** and *why* (they're hand-tuned; a clear, honest
    comparison is more impressive than a fake "I beat cuBLAS").
  - **Nsight Compute** profile: achieved occupancy, memory-bound vs compute-bound, and a
    **roofline** placing your kernel against the T4's peak bandwidth/FLOPs.
- **State the limits:** one seed / one GPU model, one problem size range, no multi-GPU, no
  fp16/tensor-core path (that's a stretch goal, not a claim).

## Deliverables (repo layout)

```
kernelforge-cuda/
  README.md               # what it is, the result table, the honesty note
  PROJECT_SPEC.md          # this file
  src/
    v0_naive.cu
    v1_shared.cu
    v2_warp.cu
    v3_topk.cu
    reference.py           # NumPy ground truth
  bench/
    run.py                 # sweeps N, B; writes results.csv
    RESULTS.md             # tables, roofline, interpretation, honest limits
  tests/
    test_correctness.py    # every kernel matches NumPy within 1e-4
  Makefile                 # nvcc build
  .github/workflows/ci.yml # builds + runs the CPU-only correctness check
```

## What NOT to claim (until true)

- Not "beat cuBLAS" — report the honest gap.
- Not tensor cores / fp16 / TensorRT unless you actually implement them.
- Not "production-scale" — it's a single-GPU study over a bounded size range.
- Not multi-GPU / NCCL.

## Environment & learning path

- **GPU:** free **Google Colab T4** (or Kaggle) — both expose `nvcc` and CUDA. Profile with
  **Nsight Compute** (`ncu`) where available; Colab supports basic profiling.
- **Learn from:** NVIDIA *CUDA C++ Programming Guide*; the book *CUDA by Example*; Mark
  Harris's classic "Optimizing Parallel Reduction" slides; and NVIDIA's GEMM-optimization
  walkthroughs. Do the reduction tutorial first (v0→v3 there mirrors this project).
- **Ramp order:** (1) reduction tutorial to learn shared memory + warp shuffle, then (2)
  this project applies the same ideas to your real embeddings.

## Realistic timeline

~**2–4 weeks** at student pace: a few days to get v0 correct, then one version every few
days, then a week on benchmarking + Nsight + writing RESULTS.md.

## Resume bullet it will earn (once real)

> Wrote a custom **CUDA** kernel for batched cosine-similarity + top-k retrieval, optimizing
> through **shared-memory tiling** and **warp-shuffle reductions** to an **Nx** speedup over a
> CPU baseline; profiled with **Nsight Compute** (roofline / occupancy) and benchmarked
> honestly against cuBLAS and PyTorch, documenting where and why the library wins.

## Stretch goals (only if the core is solid)

- fp16 / **tensor-core** path via `wmma` and compare accuracy vs speed.
- Fuse normalization into the kernel.
- Try it on an A100/L4 (Colab Pro) and compare rooflines across GPUs.
