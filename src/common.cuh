/* common.cuh — helpers shared by every kernel version. */
#ifndef KERNELFORGE_COMMON_CUH
#define KERNELFORGE_COMMON_CUH

#include <cuda_runtime.h>
#include <float.h>
#include <cstdio>
#include <chrono>

#include "kernelforge.h"
#include "topk_rule.h"

/* Bail out of the host wrapper with the failing cudaError_t. Every wrapper
 * frees through the `cleanup` label so a mid-call failure does not leak. */
#define KF_CHECK(expr)                          \
    do {                                        \
        cudaError_t _e = (expr);                \
        if (_e != cudaSuccess) {                \
            status = (int)_e;                   \
            goto cleanup;                       \
        }                                       \
    } while (0)

/* Dense-row top-k on the host, used by v0-v2 and the cuBLAS baseline. Scanning
 * i ascending and delegating every comparison to kf_insert means this matches
 * the device-side selection in v3 by construction, not by coincidence. */
static inline void kf_host_topk(const float *scores, int n, int k,
                                float *out_vals, int *out_idx) {
    kf_clear(out_vals, out_idx, k);
    for (int i = 0; i < n; ++i) kf_insert(out_vals, out_idx, k, scores[i], i);
}

static inline double kf_now_ms(void) {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

#endif /* KERNELFORGE_COMMON_CUH */
