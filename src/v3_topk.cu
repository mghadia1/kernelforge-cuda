/* v3_topk.cu — keep the top-k on the GPU.
 *
 * v0-v2 all ship B*N scores back over PCIe and select on the CPU. At N = 1M and
 * B = 32 that is 128 MB of transfer to extract 160 numbers, and on a T4 the copy
 * plus the host selection dwarf the arithmetic. This version never materializes
 * the score matrix in host memory.
 *
 * Two kernels:
 *   1. kf_v3_partial — v2's warp-per-document scoring, but each block covers a
 *      KF_CHUNK of documents, keeps their scores in shared memory, and reduces them
 *      to its own top-k. Output is gridDim.x * k candidates per query.
 *   2. kf_merge_partials (merge_topk.cuh, shared with v4) — one block per query
 *      folds those candidates into the final k.
 *
 * Both selection steps use the same (score, index) lexicographic rule as
 * reference.py: a candidate wins on a strictly higher score, or on an equal
 * score with a lower document index. That makes the result independent of the
 * order blocks happen to finish in, so the test can compare indices and not
 * just values.
 */
#include "common.cuh"
#include "v3_config.h"
#include "merge_topk.cuh"   /* kf_merge_partials, shared with v4 */

#define WARP 32
#define WARPS_PER_BLOCK (KF_V3_BLOCK / WARP)   /* 8 warps == 8 documents in flight */

static __global__ void kf_v3_partial(const float *__restrict__ q,
                                     const float *__restrict__ X,
                                     float *__restrict__ part_vals,
                                     int *__restrict__ part_idx,
                                     int N, int d, int k) {
    extern __shared__ float smem[];
    float *sQ     = smem;              /* d floats  */
    float *sScore = smem + d;          /* KF_CHUNK floats */

    const int lane   = threadIdx.x & (WARP - 1);
    const int warpId = threadIdx.x >> 5;
    const int b      = blockIdx.y;
    const int base   = blockIdx.x * KF_CHUNK;

    for (int i = threadIdx.x; i < d; i += KF_V3_BLOCK) sQ[i] = q[(size_t)b * d + i];
    __syncthreads();

    /* Warp-per-document scoring, each warp walking KF_CHUNK/WARPS_PER_BLOCK rows. */
    for (int w = warpId; w < KF_CHUNK; w += WARPS_PER_BLOCK) {
        const int n = base + w;
        float acc = 0.0f;
        if (n < N) {
            const float *xrow = X + (size_t)n * d;
            for (int i = lane; i < d; i += WARP) acc += sQ[i] * xrow[i];
            #pragma unroll
            for (int off = WARP / 2; off > 0; off >>= 1)
                acc += __shfl_down_sync(0xffffffffu, acc, off);
        }
        if (lane == 0) sScore[w] = (n < N) ? acc : -FLT_MAX;
    }
    __syncthreads();

    /* Block-level selection: 256 threads each keep a register top-k over a
     * strided slice of the chunk, then thread 0 merges the slices. */
    float vals[KF_MAX_K];
    int   idx[KF_MAX_K];
    kf_clear(vals, idx, k);
    for (int w = threadIdx.x; w < KF_CHUNK; w += KF_V3_BLOCK) {
        const int n = base + w;
        if (n < N) kf_insert(vals, idx, k, sScore[w], n);
    }

    /* sScore is reused as the candidate scratch, so every thread must be done
     * reading it before anyone writes over it. */
    __syncthreads();

    float *sVals = sScore;                       /* reuse: scores are consumed */
    int   *sIdx  = (int *)(sScore + KF_V3_BLOCK * KF_MAX_K);
    for (int r = 0; r < k; ++r) {
        sVals[threadIdx.x * k + r] = vals[r];
        sIdx[threadIdx.x * k + r]  = idx[r];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float mv[KF_MAX_K];
        int   mi[KF_MAX_K];
        kf_clear(mv, mi, k);
        for (int t = 0; t < KF_V3_BLOCK * k; ++t)
            kf_insert(mv, mi, k, sVals[t], sIdx[t]);
        const size_t out = ((size_t)b * gridDim.x + blockIdx.x) * k;
        for (int r = 0; r < k; ++r) {
            part_vals[out + r] = mv[r];
            part_idx[out + r]  = mi[r];
        }
    }
}

extern "C" int kf_v3_topk(const float *q, const float *X, int B, int N, int d,
                          int k, float *out_vals, int *out_idx,
                          KfTiming *timing) {
    int status = 0;
    float *d_q = NULL, *d_X = NULL, *d_pv = NULL, *d_ov = NULL;
    int *d_pi = NULL, *d_oi = NULL;
    cudaEvent_t ev0 = NULL, ev1 = NULL;
    double t_start = kf_now_ms(), t_h2d0, t_h2d1, t_d2h0, t_d2h1;
    float kernel_ms = 0.0f;
    const int n_part = (N + KF_CHUNK - 1) / KF_CHUNK;

    if (k > KF_MAX_K || k < 1) return -1;

    KF_CHECK(cudaMalloc(&d_q, (size_t)B * d * sizeof(float)));
    KF_CHECK(cudaMalloc(&d_X, (size_t)N * d * sizeof(float)));
    KF_CHECK(cudaMalloc(&d_pv, (size_t)B * n_part * k * sizeof(float)));
    KF_CHECK(cudaMalloc(&d_pi, (size_t)B * n_part * k * sizeof(int)));
    KF_CHECK(cudaMalloc(&d_ov, (size_t)B * k * sizeof(float)));
    KF_CHECK(cudaMalloc(&d_oi, (size_t)B * k * sizeof(int)));
    KF_CHECK(cudaEventCreate(&ev0));
    KF_CHECK(cudaEventCreate(&ev1));

    t_h2d0 = kf_now_ms();
    KF_CHECK(cudaMemcpy(d_q, q, (size_t)B * d * sizeof(float), cudaMemcpyHostToDevice));
    KF_CHECK(cudaMemcpy(d_X, X, (size_t)N * d * sizeof(float), cudaMemcpyHostToDevice));
    KF_CHECK(cudaDeviceSynchronize());
    t_h2d1 = kf_now_ms();

    {
        /* Shared memory: the query, then the chunk's scores, which are reused as
         * the per-thread candidate lists once scoring is done. */
        size_t smem = (size_t)d * sizeof(float) +
                      (size_t)KF_V3_BLOCK * KF_MAX_K * (sizeof(float) + sizeof(int));
        dim3 grid(n_part, B);
        KF_CHECK(cudaEventRecord(ev0));
        kf_v3_partial<<<grid, KF_V3_BLOCK, smem>>>(d_q, d_X, d_pv, d_pi, N, d, k);
        KF_CHECK(cudaGetLastError());
        kf_merge_partials<<<B, KF_MERGE_BLOCK>>>(d_pv, d_pi, d_ov, d_oi, n_part, k);
        KF_CHECK(cudaEventRecord(ev1));
        KF_CHECK(cudaGetLastError());
        KF_CHECK(cudaEventSynchronize(ev1));
        KF_CHECK(cudaEventElapsedTime(&kernel_ms, ev0, ev1));
    }

    t_d2h0 = kf_now_ms();
    KF_CHECK(cudaMemcpy(out_vals, d_ov, (size_t)B * k * sizeof(float), cudaMemcpyDeviceToHost));
    KF_CHECK(cudaMemcpy(out_idx, d_oi, (size_t)B * k * sizeof(int), cudaMemcpyDeviceToHost));
    t_d2h1 = kf_now_ms();

    if (timing) {
        timing->h2d_ms       = (float)(t_h2d1 - t_h2d0);
        timing->kernel_ms    = kernel_ms;
        timing->d2h_ms       = (float)(t_d2h1 - t_d2h0);
        timing->host_topk_ms = 0.0f;
        timing->total_ms     = (float)(kf_now_ms() - t_start);
    }

cleanup:
    if (ev0) cudaEventDestroy(ev0);
    if (ev1) cudaEventDestroy(ev1);
    if (d_oi) cudaFree(d_oi);
    if (d_ov) cudaFree(d_ov);
    if (d_pi) cudaFree(d_pi);
    if (d_pv) cudaFree(d_pv);
    if (d_X) cudaFree(d_X);
    if (d_q) cudaFree(d_q);
    return status;
}
