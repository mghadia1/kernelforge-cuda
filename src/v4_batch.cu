/* v4_batch.cu — tile the batch dimension so every X byte serves V4_QT queries.
 *
 * This version exists because of a measurement, not a hunch. Nsight on
 * kf_v3_partial reported 68.79% DRAM throughput against 36.83% SM throughput
 * and 4% of the T4's fp32 peak: the scoring kernel is bandwidth-bound, so the
 * only way left to go faster is to move fewer bytes.
 *
 * v3 moves far more than it needs to. One block owns one (query, chunk) pair, so
 * every query streams the whole corpus: at N = 100,000 that is 153.6 MB of X read
 * 32 times — about 4.9 GB — to answer 32 queries. The arithmetic per byte loaded
 * is one multiply-add.
 *
 * Here a block owns one chunk of documents and a tile of V4_QT queries. A warp
 * loads each X element **once** into a register and multiplies it against all
 * V4_QT query vectors held in shared memory, accumulating V4_QT running sums.
 * DRAM traffic for X drops by a factor of V4_QT; arithmetic intensity rises by
 * the same factor. This is the reuse cuBLAS gets from tiling both dimensions of
 * the GEMM, applied to the one axis this kernel was ignoring.
 *
 * Selection changes too. v3 reduced its chunk with per-thread register lists and
 * a shared-memory scratch, which at eight queries per block would cost more
 * shared memory than the scoring tile itself. Instead each warp takes one query
 * and runs k rounds of a masked, lexicographic max-reduction over its lane
 * registers with __shfl_down_sync — no shared scratch at all (select_warp.cuh,
 * shared with v5). The stage-two merge is kf_merge_partials, shared with v3.
 */
#include "common.cuh"
#include "v3_config.h"       /* KF_MAX_K, and the merge block size */
#include "v4_config.h"
#include "select_warp.cuh"
#include "merge_topk.cuh"

#define WARP 32
#define FULL_MASK KF_FULL_MASK

static __global__ void kf_v4_partial(const float *__restrict__ q,
                                     const float *__restrict__ X,
                                     float *__restrict__ part_vals,
                                     int *__restrict__ part_idx,
                                     int N, int d, int k, int B) {
    extern __shared__ float smem[];
    float *sQ     = smem;                          /* V4_QT * d */
    float *sScore = smem + (size_t)V4_QT * d;      /* V4_CHUNK * V4_QT */

    const int lane   = threadIdx.x & (WARP - 1);
    const int warpId = threadIdx.x >> 5;
    const int base   = blockIdx.x * V4_CHUNK;      /* first document of the chunk */
    const int qt0    = blockIdx.y * V4_QT;         /* first query of the tile */
    const int nq     = min(V4_QT, B - qt0);        /* tail tile may be short */

    /* Queries are contiguous rows, so this staging is fully coalesced. */
    for (int i = threadIdx.x; i < nq * d; i += V4_BLOCK)
        sQ[i] = q[(size_t)qt0 * d + i];
    __syncthreads();

    /* Scoring: one warp per document, nq accumulators per lane. Each X element is
     * read once from global memory and reused nq times out of a register. */
    for (int w = warpId; w < V4_CHUNK; w += (V4_BLOCK / WARP)) {
        const int n = base + w;
        float acc[V4_QT];
        #pragma unroll
        for (int j = 0; j < V4_QT; ++j) acc[j] = 0.0f;

        if (n < N) {
            const float *xrow = X + (size_t)n * d;
            for (int i = lane; i < d; i += WARP) {
                const float xv = xrow[i];          /* the one global load */
                for (int j = 0; j < nq; ++j) acc[j] += sQ[(size_t)j * d + i] * xv;
            }
            for (int j = 0; j < nq; ++j) {
                float a = acc[j];
                #pragma unroll
                for (int off = WARP / 2; off > 0; off >>= 1)
                    a += __shfl_down_sync(FULL_MASK, a, off);
                if (lane == 0) sScore[(size_t)w * V4_QT + j] = a;
            }
        } else if (lane == 0) {
            for (int j = 0; j < nq; ++j) sScore[(size_t)w * V4_QT + j] = -FLT_MAX;
        }
    }
    __syncthreads();

    /* Selection: warp j owns query j, k rounds of a masked lexicographic
     * max-reduction over lane registers. Shared with v5 (select_warp.cuh). */
    if (warpId < nq) {
        const size_t out = ((size_t)(qt0 + warpId) * gridDim.x + blockIdx.x) * k;
        kf_warp_select(sScore, V4_QT, warpId, V4_PER_LANE, base, N, k, lane,
                       part_vals + out, part_idx + out);
    }
}

extern "C" int kf_v4_batch(const float *q, const float *X, int B, int N, int d,
                           int k, float *out_vals, int *out_idx,
                           KfTiming *timing) {
    int status = 0;
    float *d_q = NULL, *d_X = NULL, *d_pv = NULL, *d_ov = NULL;
    int *d_pi = NULL, *d_oi = NULL;
    cudaEvent_t ev0 = NULL, ev1 = NULL;
    double t_start = kf_now_ms(), t_h2d0, t_h2d1, t_d2h0, t_d2h1;
    float kernel_ms = 0.0f;
    const int n_part = (N + V4_CHUNK - 1) / V4_CHUNK;

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
        const size_t smem = ((size_t)V4_QT * d + (size_t)V4_CHUNK * V4_QT) * sizeof(float);
        dim3 grid(n_part, (B + V4_QT - 1) / V4_QT);
        KF_CHECK(cudaEventRecord(ev0));
        kf_v4_partial<<<grid, V4_BLOCK, smem>>>(d_q, d_X, d_pv, d_pi, N, d, k, B);
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
