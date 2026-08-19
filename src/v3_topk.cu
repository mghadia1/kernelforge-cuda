/* v3_topk.cu — keep the top-k on the GPU.
 *
 * v0-v2 all ship B*N scores back over PCIe and select on the CPU. At N = 1M and
 * B = 32 that is 128 MB of transfer to extract 160 numbers, and on a T4 the copy
 * plus the host selection dwarf the arithmetic. This version never materializes
 * the score matrix in host memory.
 *
 * Two kernels:
 *   1. kf_v3_partial — v2's warp-per-document scoring, but each block covers a
 *      CHUNK of documents, keeps their scores in shared memory, and reduces them
 *      to its own top-k. Output is gridDim.x * k candidates per query.
 *   2. kf_v3_merge   — one block per query merges those candidates into the
 *      final k. Each thread keeps a register top-k over a strided slice, then
 *      thread 0 merges the 256 partial lists.
 *
 * Both selection steps use the same (score, index) lexicographic rule as
 * reference.py: a candidate wins on a strictly higher score, or on an equal
 * score with a lower document index. That makes the result independent of the
 * order blocks happen to finish in, so the test can compare indices and not
 * just values.
 */
#include "common.cuh"

#define WARP 32
#define WARPS_PER_BLOCK 8
#define V3_BLOCK (WARP * WARPS_PER_BLOCK)   /* 256 threads */
#define CHUNK 1024                          /* documents scored per block */
#define KF_MAX_K 8   /* caps shared-memory reservation; k = 5 in practice */
#define MERGE_BLOCK 256

/* Insert (s, idx) into a descending list of length k held in registers or
 * shared memory. Empty slots are marked with idx < 0. */
static __device__ __forceinline__ void kf_dev_insert(float *vals, int *idx,
                                                     int k, float s, int i) {
    if (i < 0) return;
    int last = k - 1;
    if (idx[last] >= 0 && (s < vals[last] || (s == vals[last] && i > idx[last])))
        return;
    int j = last;
    while (j > 0 && (idx[j - 1] < 0 ||
                     s > vals[j - 1] || (s == vals[j - 1] && i < idx[j - 1]))) {
        vals[j] = vals[j - 1];
        idx[j]  = idx[j - 1];
        --j;
    }
    vals[j] = s;
    idx[j]  = i;
}

static __global__ void kf_v3_partial(const float *__restrict__ q,
                                     const float *__restrict__ X,
                                     float *__restrict__ part_vals,
                                     int *__restrict__ part_idx,
                                     int N, int d, int k) {
    extern __shared__ float smem[];
    float *sQ     = smem;              /* d floats  */
    float *sScore = smem + d;          /* CHUNK floats */

    const int lane   = threadIdx.x & (WARP - 1);
    const int warpId = threadIdx.x >> 5;
    const int b      = blockIdx.y;
    const int base   = blockIdx.x * CHUNK;

    for (int i = threadIdx.x; i < d; i += V3_BLOCK) sQ[i] = q[(size_t)b * d + i];
    __syncthreads();

    /* Warp-per-document scoring, each warp walking CHUNK/WARPS_PER_BLOCK rows. */
    for (int w = warpId; w < CHUNK; w += WARPS_PER_BLOCK) {
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
    for (int r = 0; r < k; ++r) { vals[r] = -FLT_MAX; idx[r] = -1; }
    for (int w = threadIdx.x; w < CHUNK; w += V3_BLOCK) {
        const int n = base + w;
        if (n < N) kf_dev_insert(vals, idx, k, sScore[w], n);
    }

    /* sScore is reused as the candidate scratch, so every thread must be done
     * reading it before anyone writes over it. */
    __syncthreads();

    float *sVals = sScore;                       /* reuse: scores are consumed */
    int   *sIdx  = (int *)(sScore + V3_BLOCK * KF_MAX_K);
    for (int r = 0; r < k; ++r) {
        sVals[threadIdx.x * k + r] = vals[r];
        sIdx[threadIdx.x * k + r]  = idx[r];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float mv[KF_MAX_K];
        int   mi[KF_MAX_K];
        for (int r = 0; r < k; ++r) { mv[r] = -FLT_MAX; mi[r] = -1; }
        for (int t = 0; t < V3_BLOCK * k; ++t)
            kf_dev_insert(mv, mi, k, sVals[t], sIdx[t]);
        const size_t out = ((size_t)b * gridDim.x + blockIdx.x) * k;
        for (int r = 0; r < k; ++r) {
            part_vals[out + r] = mv[r];
            part_idx[out + r]  = mi[r];
        }
    }
}

static __global__ void kf_v3_merge(const float *__restrict__ part_vals,
                                   const int *__restrict__ part_idx,
                                   float *__restrict__ out_vals,
                                   int *__restrict__ out_idx,
                                   int n_part, int k) {
    __shared__ float sVals[MERGE_BLOCK * KF_MAX_K];
    __shared__ int   sIdx[MERGE_BLOCK * KF_MAX_K];

    const int b = blockIdx.x;
    const size_t base = (size_t)b * n_part * k;

    float vals[KF_MAX_K];
    int   idx[KF_MAX_K];
    for (int r = 0; r < k; ++r) { vals[r] = -FLT_MAX; idx[r] = -1; }
    for (int c = threadIdx.x; c < n_part * k; c += MERGE_BLOCK)
        kf_dev_insert(vals, idx, k, part_vals[base + c], part_idx[base + c]);

    for (int r = 0; r < k; ++r) {
        sVals[threadIdx.x * k + r] = vals[r];
        sIdx[threadIdx.x * k + r]  = idx[r];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float mv[KF_MAX_K];
        int   mi[KF_MAX_K];
        for (int r = 0; r < k; ++r) { mv[r] = -FLT_MAX; mi[r] = -1; }
        for (int t = 0; t < MERGE_BLOCK * k; ++t)
            kf_dev_insert(mv, mi, k, sVals[t], sIdx[t]);
        for (int r = 0; r < k; ++r) {
            out_vals[b * k + r] = mv[r];
            out_idx[b * k + r]  = mi[r];
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
    const int n_part = (N + CHUNK - 1) / CHUNK;

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
                      (size_t)V3_BLOCK * KF_MAX_K * (sizeof(float) + sizeof(int));
        dim3 grid(n_part, B);
        KF_CHECK(cudaEventRecord(ev0));
        kf_v3_partial<<<grid, V3_BLOCK, smem>>>(d_q, d_X, d_pv, d_pi, N, d, k);
        KF_CHECK(cudaGetLastError());
        kf_v3_merge<<<B, MERGE_BLOCK>>>(d_pv, d_pi, d_ov, d_oi, n_part, k);
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
