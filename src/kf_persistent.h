/* kf_persistent.h — the API a real retrieval service would actually use.
 *
 * Every kf_v* entry point in kernelforge.h allocates, uploads the whole corpus,
 * runs, and frees. That is the right shape for a benchmark that wants to be
 * honest about cold cost, and it is why the measured end-to-end numbers are
 * dominated by a 1.54 GB upload: at N = 1M the corpus copy is ~350 ms and the
 * scoring kernel is under 4 ms.
 *
 * No retrieval system works that way. PaperTrail uploads its embeddings once and
 * then answers thousands of queries against them. This API models that: create
 * the corpus once, then pay only for the query vectors in, the kernel, and k
 * results out.
 *
 * The two views answer different questions, and the repo reports both:
 *   kernelforge.h  — what does one cold call cost?
 *   kf_persistent.h — what does a query cost once the service is warm?
 */
#ifndef KERNELFORGE_PERSISTENT_H
#define KERNELFORGE_PERSISTENT_H

#include "kernelforge.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct KfCorpus KfCorpus;

/* Which scoring path a query uses. The early versions are deliberately absent:
 * v0-v2 ship a whole B x N score matrix back to the host, so a persistent
 * corpus does not change what they are. */
enum {
    KF_IMPL_V3 = 0,   /* on-GPU top-k */
    KF_IMPL_V4 = 1,   /* batch tiling */
    KF_IMPL_V5 = 2,   /* register blocking */
    KF_IMPL_CUBLAS = 3
};

/* Uploads X (N x d, row-major) to the device and keeps it there. Returns NULL
 * on failure, with the cudaError_t in *status if status is non-NULL. */
KfCorpus *kf_corpus_create(const float *X, int N, int d, int *status);

/* One query batch against the resident corpus. Scratch buffers are allocated on
 * first use and reused after that, so a steady-state query allocates nothing.
 * timing->h2d_ms covers only the query vectors; total_ms is the whole call. */
int kf_corpus_query(KfCorpus *c, int impl, const float *q, int B, int k,
                    float *out_vals, int *out_idx, KfTiming *timing);

void kf_corpus_destroy(KfCorpus *c);

#ifdef __cplusplus
}
#endif
#endif
