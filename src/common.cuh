/* common.cuh — helpers shared by every kernel version. */
#ifndef KERNELFORGE_COMMON_CUH
#define KERNELFORGE_COMMON_CUH

#include <cuda_runtime.h>
#include <float.h>
#include <cstdio>
#include <chrono>

#include "kernelforge.h"

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

/* Single-pass insertion top-k over a dense score row. k is 5 in practice, so a
 * sorted k-element list is cheaper than sorting n scores and it makes the
 * tie-break rule explicit: a candidate must be strictly greater to displace an
 * incumbent, so when scores tie the lower index survives (we scan i ascending).
 * reference.py uses the same rule, which is what lets the tests compare indices
 * and not just values. */
static inline void kf_host_topk(const float *scores, int n, int k,
                                float *out_vals, int *out_idx) {
    for (int r = 0; r < k; ++r) {
        out_vals[r] = -FLT_MAX;
        out_idx[r]  = -1;
    }
    for (int i = 0; i < n; ++i) {
        float s = scores[i];
        if (s <= out_vals[k - 1] && out_idx[k - 1] >= 0) continue;
        int j = k - 1;
        while (j > 0 && (out_idx[j - 1] < 0 || s > out_vals[j - 1])) {
            out_vals[j] = out_vals[j - 1];
            out_idx[j]  = out_idx[j - 1];
            --j;
        }
        out_vals[j] = s;
        out_idx[j]  = i;
    }
}

static inline double kf_now_ms(void) {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

#endif /* KERNELFORGE_COMMON_CUH */
