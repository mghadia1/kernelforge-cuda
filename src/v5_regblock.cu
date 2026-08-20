/* v5_regblock.cu — register-block the query tile.
 *
 * Chosen from a measurement, again. After v4 the profile said the kernel was no
 * longer DRAM-bound (39.27%) but that L1/TEX throughput had become the highest
 * utilization in the kernel at 78.48%. That number has a specific cause: v4's
 * inner step loads one X element and then reads all eight query values back out
 * of shared memory to multiply against it, so shared traffic scales with the
 * number of FMAs.
 *
 * v5 turns the loop inside out. A warp now owns V5_DR documents at once. Each
 * step loads the eight query values **once into registers**, then loads V5_DR
 * X elements and multiplies each against the registers:
 *
 *     v4 per step:  1 global load  +  QT shared loads  ->  QT FMAs
 *     v5 per step:  DR global loads +  QT shared loads  ->  DR*QT FMAs
 *
 * Shared-memory traffic per FMA falls by a factor of V5_DR, at the cost of
 * DR*QT accumulators live in registers (4x8 = 32 floats, plus 8 for the query
 * values). This is the second half of what a tuned GEMM does; v4 was the first
 * half.
 *
 * The tail queries are handled by zero-filling the unused rows of the query
 * tile, which keeps every inner loop bound a compile-time constant so the
 * accumulators stay in registers instead of spilling to local memory. Scores
 * for those padded columns are computed and never read.
 *
 * Selection and the merge are unchanged from v4 — same routine, same geometry,
 * so the existing host-side simulation covers them.
 */
#include "common.cuh"
#include "v3_config.h"        /* KF_MAX_K, KF_MERGE_BLOCK */
#include "v5_config.h"
#include "select_warp.cuh"
#include "merge_topk.cuh"

#define WARP 32
#define V5_WARPS (V5_BLOCK / WARP)

static __global__ void kf_v5_partial(const float *__restrict__ q,
                                     const float *__restrict__ X,
                                     float *__restrict__ part_vals,
                                     int *__restrict__ part_idx,
                                     int N, int d, int k, int B) {
    extern __shared__ float smem[];
    float *sQ     = smem;                          /* V5_QT * d */
    float *sScore = smem + (size_t)V5_QT * d;      /* V5_CHUNK * V5_QT */

    const int lane   = threadIdx.x & (WARP - 1);
    const int warpId = threadIdx.x >> 5;
    const int base   = blockIdx.x * V5_CHUNK;
    const int qt0    = blockIdx.y * V5_QT;
    const int nq     = min(V5_QT, B - qt0);

    /* Stage the query tile, zero-filling rows past the end of the batch so the
     * inner loops below can run to a compile-time V5_QT. */
    for (int i = threadIdx.x; i < V5_QT * d; i += V5_BLOCK) {
        const int j = i / d;
        sQ[i] = (j < nq) ? q[(size_t)qt0 * d + i] : 0.0f;
    }
    __syncthreads();

    for (int w0 = warpId * V5_DR; w0 < V5_CHUNK; w0 += V5_WARPS * V5_DR) {
        float acc[V5_DR][V5_QT];
        #pragma unroll
        for (int r = 0; r < V5_DR; ++r)
            #pragma unroll
            for (int j = 0; j < V5_QT; ++j) acc[r][j] = 0.0f;

        for (int i = lane; i < d; i += WARP) {
            /* The whole point: eight shared-memory reads, then DR*QT FMAs. */
            float qv[V5_QT];
            #pragma unroll
            for (int j = 0; j < V5_QT; ++j) qv[j] = sQ[(size_t)j * d + i];

            #pragma unroll
            for (int r = 0; r < V5_DR; ++r) {
                const int n = base + w0 + r;
                if (n < N) {
                    const float xv = X[(size_t)n * d + i];
                    #pragma unroll
                    for (int j = 0; j < V5_QT; ++j) acc[r][j] += qv[j] * xv;
                }
            }
        }

        #pragma unroll
        for (int r = 0; r < V5_DR; ++r) {
            const int n = base + w0 + r;
            for (int j = 0; j < nq; ++j) {
                float a = acc[r][j];
                #pragma unroll
                for (int off = WARP / 2; off > 0; off >>= 1)
                    a += __shfl_down_sync(KF_FULL_MASK, a, off);
                if (lane == 0)
                    sScore[(size_t)(w0 + r) * V5_QT + j] = (n < N) ? a : -FLT_MAX;
            }
        }
    }
    __syncthreads();

    if (warpId < nq) {
        const size_t out = ((size_t)(qt0 + warpId) * gridDim.x + blockIdx.x) * k;
        kf_warp_select(sScore, V5_QT, warpId, V5_PER_LANE, base, N, k, lane,
                       part_vals + out, part_idx + out);
    }
}

extern "C" int kf_v5_regblock(const float *q, const float *X, int B, int N, int d,
                              int k, float *out_vals, int *out_idx,
                              KfTiming *timing) {
    int status = 0;
    float *d_q = NULL, *d_X = NULL, *d_pv = NULL, *d_ov = NULL;
    int *d_pi = NULL, *d_oi = NULL;
    cudaEvent_t ev0 = NULL, ev1 = NULL;
    double t_start = kf_now_ms(), t_h2d0, t_h2d1, t_d2h0, t_d2h1;
    float kernel_ms = 0.0f;
    const int n_part = (N + V5_CHUNK - 1) / V5_CHUNK;

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
        const size_t smem = ((size_t)V5_QT * d + (size_t)V5_CHUNK * V5_QT) * sizeof(float);
        dim3 grid(n_part, (B + V5_QT - 1) / V5_QT);
        KF_CHECK(cudaEventRecord(ev0));
        kf_v5_partial<<<grid, V5_BLOCK, smem>>>(d_q, d_X, d_pv, d_pi, N, d, k, B);
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
