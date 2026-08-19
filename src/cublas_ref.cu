/* cublas_ref.cu — the library baseline this project is measured against.
 *
 * Same problem, same host wrapper, same timing points as the hand-written
 * kernels, but the scoring is one cublasSgemm and the top-k runs on the host.
 * Keeping it behind the identical C ABI is the point: the benchmark compares
 * like with like, and the honest gap between kf_v3_topk and this is a result,
 * not a failure.
 *
 * Layout note: cuBLAS is column-major. A row-major (N x d) buffer *is* a
 * column-major (d x N) matrix, so scores = X @ q^T becomes op(A) = A^T with
 * lda = d, and the column-major (N x B) output is exactly the row-major (B x N)
 * score matrix the rest of the code expects.
 */
#include <cublas_v2.h>

#include "common.cuh"

extern "C" int kf_cublas(const float *q, const float *X, int B, int N, int d,
                         int k, float *out_vals, int *out_idx,
                         KfTiming *timing) {
    int status = 0;
    float *d_q = NULL, *d_X = NULL, *d_scores = NULL, *h_scores = NULL;
    cudaEvent_t ev0 = NULL, ev1 = NULL;
    cublasHandle_t handle = NULL;
    double t_start = kf_now_ms(), t_h2d0, t_h2d1, t_d2h0, t_d2h1, t_topk0, t_topk1;
    float kernel_ms = 0.0f;
    const float alpha = 1.0f, beta = 0.0f;

    KF_CHECK(cudaMalloc(&d_q, (size_t)B * d * sizeof(float)));
    KF_CHECK(cudaMalloc(&d_X, (size_t)N * d * sizeof(float)));
    KF_CHECK(cudaMalloc(&d_scores, (size_t)B * N * sizeof(float)));
    KF_CHECK(cudaMallocHost(&h_scores, (size_t)B * N * sizeof(float)));
    KF_CHECK(cudaEventCreate(&ev0));
    KF_CHECK(cudaEventCreate(&ev1));
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { status = -2; goto cleanup; }

    t_h2d0 = kf_now_ms();
    KF_CHECK(cudaMemcpy(d_q, q, (size_t)B * d * sizeof(float), cudaMemcpyHostToDevice));
    KF_CHECK(cudaMemcpy(d_X, X, (size_t)N * d * sizeof(float), cudaMemcpyHostToDevice));
    KF_CHECK(cudaDeviceSynchronize());
    t_h2d1 = kf_now_ms();

    KF_CHECK(cudaEventRecord(ev0));
    if (cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, B, d,
                    &alpha, d_X, d, d_q, d, &beta, d_scores, N)
        != CUBLAS_STATUS_SUCCESS) {
        status = -3;
        goto cleanup;
    }
    KF_CHECK(cudaEventRecord(ev1));
    KF_CHECK(cudaEventSynchronize(ev1));
    KF_CHECK(cudaEventElapsedTime(&kernel_ms, ev0, ev1));

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
    if (handle) cublasDestroy(handle);
    if (ev0) cudaEventDestroy(ev0);
    if (ev1) cudaEventDestroy(ev1);
    if (h_scores) cudaFreeHost(h_scores);
    if (d_scores) cudaFree(d_scores);
    if (d_X) cudaFree(d_X);
    if (d_q) cudaFree(d_q);
    return status;
}
