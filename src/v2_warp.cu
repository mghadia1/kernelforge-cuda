/* v2_warp.cu — one warp per (query, document), warp-shuffle reduction.
 *
 * v1 got coalescing back by paying for a shared-memory round trip: every X
 * element is written to shared and read again, and the block has to synchronize
 * once per d-tile. For a dot product that is more machinery than the problem
 * needs.
 *
 * Here a whole warp owns one document. Lane l reads X[n][l], X[n][l+32], ...
 * straight from global memory, which is already perfectly coalesced — 32 lanes
 * covering 128 contiguous bytes — so X never touches shared memory at all. The
 * per-lane partials collapse with __shfl_down_sync, a register-to-register
 * exchange that needs no shared memory and no __syncthreads.
 *
 * The query vector is the one thing every warp in the block re-reads, so it is
 * staged in shared memory once per block (d floats, 1.5 KB at d = 384).
 *
 * Top-k still runs on the host; that is v3's job.
 */
#include "common.cuh"

#define WARP 32
#define WARPS_PER_BLOCK 8                       /* 256 threads == 8 documents */
#define V2_BLOCK (WARP * WARPS_PER_BLOCK)

static __global__ void kf_v2_scores(const float *__restrict__ q,
                                    const float *__restrict__ X,
                                    float *__restrict__ scores,
                                    int N, int d) {
    extern __shared__ float sQ[];               /* d floats */

    const int lane   = threadIdx.x & (WARP - 1);
    const int warpId = threadIdx.x >> 5;
    const int b      = blockIdx.y;
    const int n      = blockIdx.x * WARPS_PER_BLOCK + warpId;

    for (int i = threadIdx.x; i < d; i += V2_BLOCK) sQ[i] = q[(size_t)b * d + i];
    __syncthreads();

    if (n >= N) return;

    const float *xrow = X + (size_t)n * d;
    float acc = 0.0f;
    for (int i = lane; i < d; i += WARP) acc += sQ[i] * xrow[i];

    /* Tree reduction inside the warp: after five steps lane 0 has the sum. */
    #pragma unroll
    for (int off = WARP / 2; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);

    if (lane == 0) scores[(size_t)b * N + n] = acc;
}

extern "C" int kf_v2_warp(const float *q, const float *X, int B, int N, int d,
                          int k, float *out_vals, int *out_idx,
                          KfTiming *timing) {
    int status = 0;
    float *d_q = NULL, *d_X = NULL, *d_scores = NULL, *h_scores = NULL;
    cudaEvent_t ev0 = NULL, ev1 = NULL;
    double t_start = kf_now_ms(), t_h2d0, t_h2d1, t_d2h0, t_d2h1, t_topk0, t_topk1;
    float kernel_ms = 0.0f;

    KF_CHECK(cudaMalloc(&d_q, (size_t)B * d * sizeof(float)));
    KF_CHECK(cudaMalloc(&d_X, (size_t)N * d * sizeof(float)));
    KF_CHECK(cudaMalloc(&d_scores, (size_t)B * N * sizeof(float)));
    KF_CHECK(cudaMallocHost(&h_scores, (size_t)B * N * sizeof(float)));
    KF_CHECK(cudaEventCreate(&ev0));
    KF_CHECK(cudaEventCreate(&ev1));

    t_h2d0 = kf_now_ms();
    KF_CHECK(cudaMemcpy(d_q, q, (size_t)B * d * sizeof(float), cudaMemcpyHostToDevice));
    KF_CHECK(cudaMemcpy(d_X, X, (size_t)N * d * sizeof(float), cudaMemcpyHostToDevice));
    KF_CHECK(cudaDeviceSynchronize());
    t_h2d1 = kf_now_ms();

    {
        dim3 block(V2_BLOCK);
        dim3 grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, B);
        size_t smem = (size_t)d * sizeof(float);
        KF_CHECK(cudaEventRecord(ev0));
        kf_v2_scores<<<grid, block, smem>>>(d_q, d_X, d_scores, N, d);
        KF_CHECK(cudaEventRecord(ev1));
        KF_CHECK(cudaGetLastError());
        KF_CHECK(cudaEventSynchronize(ev1));
        KF_CHECK(cudaEventElapsedTime(&kernel_ms, ev0, ev1));
    }

    t_d2h0 = kf_now_ms();
    KF_CHECK(cudaMemcpy(h_scores, d_scores, (size_t)B * N * sizeof(float),
                        cudaMemcpyDeviceToHost));
    t_d2h1 = kf_now_ms();

    t_topk0 = kf_now_ms();
    for (int b = 0; b < B; ++b)
        kf_host_topk(h_scores + (size_t)b * N, N, k, out_vals + b * k, out_idx + b * k);
    t_topk1 = kf_now_ms();

    if (timing) {
        timing->h2d_ms       = (float)(t_h2d1 - t_h2d0);
        timing->kernel_ms    = kernel_ms;
        timing->d2h_ms       = (float)(t_d2h1 - t_d2h0);
        timing->host_topk_ms = (float)(t_topk1 - t_topk0);
        timing->total_ms     = (float)(kf_now_ms() - t_start);
    }

cleanup:
    if (ev0) cudaEventDestroy(ev0);
    if (ev1) cudaEventDestroy(ev1);
    if (h_scores) cudaFreeHost(h_scores);
    if (d_scores) cudaFree(d_scores);
    if (d_X) cudaFree(d_X);
    if (d_q) cudaFree(d_q);
    return status;
}
