/* v1_shared.cu — shared-memory tiling.
 *
 * v0's problem is the access pattern, not the arithmetic: neighbouring threads
 * read addresses d floats apart, so every warp load touches 32 different cache
 * lines. Here a block cooperatively stages a tile of X (TILE_N documents x
 * TILE_D dimensions) into shared memory using *coalesced* global reads —
 * consecutive threads read consecutive dimensions of the same row — and then
 * each thread consumes its own document's row out of shared memory.
 *
 * The tile row stride is padded to TILE_D + 1 so that the compute phase, where
 * thread t reads sX[t][i] for a fixed i, spreads across all 32 banks instead of
 * hitting one. Without the pad this phase is a 32-way bank conflict and the
 * tiling wins back almost nothing.
 *
 * Top-k still runs on the host; that is v3's job.
 */
#include "common.cuh"

#define TILE_N 128   /* documents per block == threads per block */
#define TILE_D 32    /* dimensions staged per iteration */
#define SROW   (TILE_D + 1)

static __global__ void kf_v1_scores(const float *__restrict__ q,
                                    const float *__restrict__ X,
                                    float *__restrict__ scores,
                                    int N, int d) {
    __shared__ float sX[TILE_N * SROW];
    __shared__ float sQ[TILE_D];

    const int tid = threadIdx.x;
    const int n0  = blockIdx.x * TILE_N;          /* first document of the tile */
    const int b   = blockIdx.y;                   /* query */
    const int n   = n0 + tid;                     /* this thread's document */
    const int rows = min(TILE_N, N - n0);         /* tail block may be short */

    const float *qrow = q + (size_t)b * d;
    float acc = 0.0f;

    for (int d0 = 0; d0 < d; d0 += TILE_D) {
        const int dims = min(TILE_D, d - d0);

        if (tid < dims) sQ[tid] = qrow[d0 + tid];

        /* Coalesced staging: element index -> (row, dim) with dim fastest, so
         * consecutive threads hit consecutive addresses inside one X row. */
        for (int idx = tid; idx < rows * dims; idx += TILE_N) {
            const int r = idx / dims;
            const int c = idx - r * dims;
            sX[r * SROW + c] = X[(size_t)(n0 + r) * d + d0 + c];
        }
        __syncthreads();

        if (n < N) {
            const float *srow = sX + tid * SROW;
            for (int i = 0; i < dims; ++i) acc += sQ[i] * srow[i];
        }
        __syncthreads();
    }

    if (n < N) scores[(size_t)b * N + n] = acc;
}

extern "C" int kf_v1_shared(const float *q, const float *X, int B, int N, int d,
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
        dim3 block(TILE_N);
        dim3 grid((N + TILE_N - 1) / TILE_N, B);
        KF_CHECK(cudaEventRecord(ev0));
        kf_v1_scores<<<grid, block>>>(d_q, d_X, d_scores, N, d);
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
