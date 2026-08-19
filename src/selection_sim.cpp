/* selection_sim.cpp — v3's two-stage top-k, executed serially on the CPU.
 *
 * Why this exists: v3 is the only version whose *answer* depends on how work is
 * split across blocks and threads. v0-v2 produce a plain score matrix and select
 * on the host; v3 selects in two stages, and a mistake in either — a wrong
 * stride, a candidate dropped at a chunk boundary, a tie resolved by arrival
 * order — produces a plausible-looking wrong answer.
 *
 * There is no GPU on the development machine, so that logic would otherwise go
 * untested until it ran on a T4. This file runs the identical decomposition
 * (same KF_CHUNK, same 256-thread strides, same per-block merge, same final
 * merge) against the same kf_insert rule, in plain C++, so tests/test_selection.py
 * can hammer it with adversarial inputs — heavy ties, N on and off chunk
 * boundaries, k = 1 and k = KF_MAX_K — on any machine.
 *
 * What it does NOT test: the CUDA itself. Shared-memory reuse hazards, the
 * missing __syncthreads, occupancy, launch configuration, and every arithmetic
 * detail of the dot product are invisible here. This narrows where a bug can
 * hide; it does not prove the kernel correct. Only the T4 run does that.
 *
 * Built by `make sim` with a plain host compiler, no CUDA required.
 */
#include <cstdlib>
#include <vector>

#include "topk_rule.h"
#include "v3_config.h"

/* One block of kf_v3_partial, minus the scoring: KF_V3_BLOCK "threads" each keep
 * a top-k over a strided slice of the chunk, then thread 0 merges the slices. */
static void sim_partial_block(const float *scores, int N, int base, int k,
                              float *out_vals, int *out_idx) {
    std::vector<float> tvals((size_t)KF_V3_BLOCK * k);
    std::vector<int>   tidx((size_t)KF_V3_BLOCK * k);

    for (int t = 0; t < KF_V3_BLOCK; ++t) {
        float *v = &tvals[(size_t)t * k];
        int   *i = &tidx[(size_t)t * k];
        kf_clear(v, i, k);
        for (int w = t; w < KF_CHUNK; w += KF_V3_BLOCK) {
            const int n = base + w;
            if (n < N) kf_insert(v, i, k, scores[n], n);
        }
    }

    kf_clear(out_vals, out_idx, k);
    for (int t = 0; t < KF_V3_BLOCK * k; ++t)
        kf_insert(out_vals, out_idx, k, tvals[t], tidx[t]);
}

/* kf_v3_merge for one query: strided register lists, then a final merge. */
static void sim_merge(const float *pv, const int *pi, int n_cand, int k,
                      float *out_vals, int *out_idx) {
    std::vector<float> tvals((size_t)KF_MERGE_BLOCK * k);
    std::vector<int>   tidx((size_t)KF_MERGE_BLOCK * k);

    for (int t = 0; t < KF_MERGE_BLOCK; ++t) {
        float *v = &tvals[(size_t)t * k];
        int   *i = &tidx[(size_t)t * k];
        kf_clear(v, i, k);
        for (int c = t; c < n_cand; c += KF_MERGE_BLOCK)
            kf_insert(v, i, k, pv[c], pi[c]);
    }

    kf_clear(out_vals, out_idx, k);
    for (int t = 0; t < KF_MERGE_BLOCK * k; ++t)
        kf_insert(out_vals, out_idx, k, tvals[t], tidx[t]);
}

/* Full pipeline over a B x N score matrix. Returns 0, or -1 for an out-of-range
 * k — the same guard kf_v3_topk applies before it allocates anything. */
extern "C" int kf_sim_v3_select(const float *scores, int B, int N, int k,
                                float *out_vals, int *out_idx) {
    if (k < 1 || k > KF_MAX_K) return -1;
    const int n_part = (N + KF_CHUNK - 1) / KF_CHUNK;

    std::vector<float> pv((size_t)n_part * k);
    std::vector<int>   pi((size_t)n_part * k);

    for (int b = 0; b < B; ++b) {
        const float *row = scores + (size_t)b * N;
        for (int p = 0; p < n_part; ++p)
            sim_partial_block(row, N, p * KF_CHUNK, k,
                              &pv[(size_t)p * k], &pi[(size_t)p * k]);
        sim_merge(pv.data(), pi.data(), n_part * k, k,
                  out_vals + (size_t)b * k, out_idx + (size_t)b * k);
    }
    return 0;
}

/* Exposed so a test can pin the rule itself, independent of the pipeline. */
extern "C" int kf_sim_insert_rule(float s, int i, float v, int j) {
    return kf_outranks(s, i, v, j) ? 1 : 0;
}
