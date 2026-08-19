/* kernelforge.h — C ABI shared by every kernel version.
 *
 * Problem: given B query vectors q (row-major, B x d) and a corpus X (row-major,
 * N x d), all L2-normalized, compute the top-k highest cosine similarities per
 * query. Because the inputs are normalized, cosine similarity is a dot product.
 *
 * Every kf_v* function has the same signature so bench/run.py can sweep them
 * uniformly. Outputs are B x k, sorted descending by score; ties are broken by
 * the lower document index so results match reference.py exactly.
 */
#ifndef KERNELFORGE_H
#define KERNELFORGE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Milliseconds. kernel_ms is measured with CUDA events around the kernel(s)
 * only; total_ms is wall time for the whole call including transfers. */
typedef struct {
    float h2d_ms;
    float kernel_ms;
    float d2h_ms;
    float host_topk_ms;
    float total_ms;
} KfTiming;

/* All return 0 on success, or the cudaError_t value that failed.
 * out_vals: B*k floats. out_idx: B*k ints. timing may be NULL. */
int kf_v0_naive (const float *q, const float *X, int B, int N, int d, int k,
                 float *out_vals, int *out_idx, KfTiming *timing);
int kf_v1_shared(const float *q, const float *X, int B, int N, int d, int k,
                 float *out_vals, int *out_idx, KfTiming *timing);
int kf_v2_warp  (const float *q, const float *X, int B, int N, int d, int k,
                 float *out_vals, int *out_idx, KfTiming *timing);
int kf_v3_topk  (const float *q, const float *X, int B, int N, int d, int k,
                 float *out_vals, int *out_idx, KfTiming *timing);

/* Library baseline: cublasSgemm scoring + host top-k, same ABI and same timing
 * points, so the comparison in bench/RESULTS.md is like for like. */
int kf_cublas   (const float *q, const float *X, int B, int N, int d, int k,
                 float *out_vals, int *out_idx, KfTiming *timing);

/* Environment probes, so the Python side can report what it actually ran on. */
int kf_device_count(void);
int kf_device_name(char *buf, int len);   /* device 0; 0 on success */

#ifdef __cplusplus
}
#endif
#endif /* KERNELFORGE_H */
