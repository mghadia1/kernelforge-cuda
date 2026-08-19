/* v0_naive.cu — the correctness baseline.
 *
 * One thread per (query, document) pair. Each thread walks the whole d-length
 * row of X on its own and accumulates a dot product. Nothing is cached and
 * nothing is coalesced: adjacent threads handle adjacent *documents*, so at
 * every step of the inner loop they read addresses d floats apart. That is the
 * worst possible access pattern for a memory-bound problem, which is exactly
 * why this version exists — it is the number the later versions have to beat.
 *
 * Top-k runs on the host, so all B*N scores are shipped back over PCIe.
 */
#include "common.cuh"

static __global__ void kf_v0_scores(const float *__restrict__ q,
                                    const float *__restrict__ X,
                                    float *__restrict__ scores,
                                    int N, int d) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;   /* document */
    int b = blockIdx.y;                              /* query */
    if (n >= N) return;

    const float *xrow = X + (size_t)n * d;
    const float *qrow = q + (size_t)b * d;
    float acc = 0.0f;
    for (int i = 0; i < d; ++i) acc += qrow[i] * xrow[i];

    scores[(size_t)b * N + n] = acc;
}

extern "C" int kf_v0_naive(const float *q, const float *X, int B, int N, int d,
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
        dim3 block(256);
        dim3 grid((N + block.x - 1) / block.x, B);
        KF_CHECK(cudaEventRecord(ev0));
        kf_v0_scores<<<grid, block>>>(d_q, d_X, d_scores, N, d);
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

extern "C" int kf_device_count(void) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n;
}

extern "C" int kf_device_name(char *buf, int len) {
    cudaDeviceProp prop;
    cudaError_t e = cudaGetDeviceProperties(&prop, 0);
    if (e != cudaSuccess) return (int)e;
    snprintf(buf, len, "%s (sm_%d%d, %.1f GB)", prop.name, prop.major, prop.minor,
             (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    return 0;
}
