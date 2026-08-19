/* topk_rule.h — the one selection rule, shared by host, device, and the CPU
 * simulation in selection_sim.cpp.
 *
 * There is exactly one comparison in this project, and it lives here. Every
 * place that picks a top-k — the host fallback in common.cuh, the per-block and
 * merge stages of v3, and the CPU simulation the tests exercise without a GPU —
 * calls kf_insert, so they cannot drift apart.
 *
 * The rule: a candidate (score s, document i) beats an incumbent (v, j) when
 * s > v, or when s == v and i < j. Ties going to the lower index is what makes
 * the answer independent of the order blocks finish in, and it is the same rule
 * numpy's stable argsort gives reference.py. Without the index half of the
 * comparison, two documents with identical scores would resolve differently on
 * every run and the tests could only ever compare values.
 *
 * Empty slots are marked idx < 0 and lose to everything.
 */
#ifndef KERNELFORGE_TOPK_RULE_H
#define KERNELFORGE_TOPK_RULE_H

#ifdef __CUDACC__
#define KF_HD __host__ __device__ __forceinline__
#else
#define KF_HD inline
#endif

/* True if (s, i) outranks (v, j). */
KF_HD bool kf_outranks(float s, int i, float v, int j) {
    if (j < 0) return true;            /* empty slot */
    if (i < 0) return false;           /* empty candidate */
    return s > v || (s == v && i < j);
}

/* Insert (s, i) into a descending list of length k. vals/idx may live in
 * registers, shared memory, or plain host memory. */
KF_HD void kf_insert(float *vals, int *idx, int k, float s, int i) {
    if (i < 0) return;
    const int last = k - 1;
    if (!kf_outranks(s, i, vals[last], idx[last])) return;
    int j = last;
    while (j > 0 && kf_outranks(s, i, vals[j - 1], idx[j - 1])) {
        vals[j] = vals[j - 1];
        idx[j]  = idx[j - 1];
        --j;
    }
    vals[j] = s;
    idx[j]  = i;
}

/* Initialize an empty list. */
KF_HD void kf_clear(float *vals, int *idx, int k) {
    for (int r = 0; r < k; ++r) { vals[r] = -3.0e38f; idx[r] = -1; }
}

#endif /* KERNELFORGE_TOPK_RULE_H */
