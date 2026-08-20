/* select_warp.cuh — v4's and v5's shared chunk selection.
 *
 * One warp owns one query. Each lane holds V*_PER_LANE of the chunk's scores;
 * k rounds of a masked lexicographic max-reduction with __shfl_down_sync pull
 * out the top k with no shared-memory scratch at all — which is what makes it
 * affordable to keep eight queries in flight per block.
 *
 * The used-bit mask is the part that has to be right: without it a lane would
 * keep re-offering the document it already won, and the kernel would return the
 * same index k times. tests/test_selection.py plants every winner inside one
 * lane's entries specifically to catch that.
 *
 * Kept in a header so v4 and v5 run the identical routine and the CPU
 * simulation of it stays valid for both.
 */
#ifndef KERNELFORGE_SELECT_WARP_CUH
#define KERNELFORGE_SELECT_WARP_CUH

#include "common.cuh"

#define KF_FULL_MASK 0xffffffffu

/* sScore is laid out [chunk][qt_stride]; this selects over column `j`.
 * Only lane 0 writes, into dst_vals[0..k) / dst_idx[0..k). */
static __device__ __forceinline__ void kf_warp_select(const float *sScore, int qt_stride,
                                                      int j, int per_lane, int base,
                                                      int N, int k, int lane,
                                                      float *dst_vals, int *dst_idx) {
    unsigned used = 0u;

    for (int r = 0; r < k; ++r) {
        float bv = -FLT_MAX;
        int   bi = -1, be = -1;

        for (int e = 0; e < per_lane; ++e) {
            if (used & (1u << e)) continue;
            const int w = e * 32 + lane;
            const int n = base + w;
            if (n >= N) continue;
            const float s = sScore[(size_t)w * qt_stride + j];
            if (kf_outranks(s, n, bv, bi)) { bv = s; bi = n; be = e; }
        }

        /* Keep this lane's own candidate: the reduction overwrites bv/bi. */
        const int my_bi = bi, my_be = be;

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(KF_FULL_MASK, bv, off);
            const int   oi = __shfl_down_sync(KF_FULL_MASK, bi, off);
            if (kf_outranks(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        const float wv = __shfl_sync(KF_FULL_MASK, bv, 0);
        const int   wi = __shfl_sync(KF_FULL_MASK, bi, 0);

        /* Document indices are unique, so exactly one lane matches. */
        if (my_be >= 0 && my_bi == wi) used |= (1u << my_be);

        if (lane == 0) {
            dst_vals[r] = (wi < 0) ? -FLT_MAX : wv;
            dst_idx[r]  = wi;
        }
    }
}

#endif
