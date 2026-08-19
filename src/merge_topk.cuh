/* merge_topk.cuh — the second stage of the two-stage top-k, shared by v3 and v4.
 *
 * Both kernels produce the same thing in stage one: n_part candidate lists of k
 * entries per query. Folding those into the final k is identical work, so it is
 * written once here rather than copied into each file where the two could drift.
 *
 * One block per query. Each thread keeps a register top-k over a strided slice of
 * the candidates, then thread 0 folds the 256 partial lists. Every comparison
 * goes through kf_insert, so ties resolve by lower document index no matter which
 * block produced the candidate.
 */
#ifndef KERNELFORGE_MERGE_TOPK_CUH
#define KERNELFORGE_MERGE_TOPK_CUH

#include "common.cuh"
#include "v3_config.h"

static __global__ void kf_merge_partials(const float *__restrict__ part_vals,
                                         const int *__restrict__ part_idx,
                                         float *__restrict__ out_vals,
                                         int *__restrict__ out_idx,
                                         int n_part, int k) {
    __shared__ float sVals[KF_MERGE_BLOCK * KF_MAX_K];
    __shared__ int   sIdx[KF_MERGE_BLOCK * KF_MAX_K];

    const int b = blockIdx.x;
    const size_t base = (size_t)b * n_part * k;

    float vals[KF_MAX_K];
    int   idx[KF_MAX_K];
    kf_clear(vals, idx, k);
    for (int c = threadIdx.x; c < n_part * k; c += KF_MERGE_BLOCK)
        kf_insert(vals, idx, k, part_vals[base + c], part_idx[base + c]);

    for (int r = 0; r < k; ++r) {
        sVals[threadIdx.x * k + r] = vals[r];
        sIdx[threadIdx.x * k + r]  = idx[r];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float mv[KF_MAX_K];
        int   mi[KF_MAX_K];
        kf_clear(mv, mi, k);
        for (int t = 0; t < KF_MERGE_BLOCK * k; ++t)
            kf_insert(mv, mi, k, sVals[t], sIdx[t]);
        for (int r = 0; r < k; ++r) {
            out_vals[b * k + r] = mv[r];
            out_idx[b * k + r]  = mi[r];
        }
    }
}

#endif
