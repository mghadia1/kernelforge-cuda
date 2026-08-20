/* persistent.cu — the corpus stays on the device between queries.
 *
 * See kf_persistent.h for why this exists. In one line: the cold-path numbers
 * are dominated by a 1.54 GB upload that a real service performs once, which
 * makes a scoring kernel that got 1.91x faster look like a 4% improvement.
 *
 * Everything here is deliberately boring. The corpus is uploaded in create();
 * the query path allocates scratch on first use, reuses it afterwards, and grows
 * it only if a later query needs more room — so a steady-state query performs no
 * allocation at all. The scoring itself calls the same kf_launch_* functions the
 * cold path uses, so the two views cannot diverge into measuring different code.
 */
#include <cstdlib>

#include "common.cuh"
#include "kf_persistent.h"
#include "v3_config.h"
#include "v4_config.h"
#include "v5_config.h"

extern "C" {
int kf_launch_v3(const float *, const float *, int, int, int, int, float *, int *, float *, int *);
int kf_launch_v4(const float *, const float *, int, int, int, int, float *, int *, float *, int *);
int kf_launch_v5(const float *, const float *, int, int, int, int, float *, int *, float *, int *);
int kf_launch_cublas(const float *, const float *, int, int, int, float *);
}

struct KfCorpus {
    float *d_X;
    int N, d;

    /* Every buffer carries its own capacity. Sharing one counter between the
     * value and index arrays looks tidy and is a null-pointer bug: reserving
     * the first would mark the pair as satisfied and the second would never be
     * allocated. */
    float *d_q;   int q_cap;
    float *d_pv;  int pv_cap;
    int   *d_pi;  int pi_cap;
    float *d_ov;  int ov_cap;
    int   *d_oi;  int oi_cap;
    float *d_scores; size_t score_cap;               /* cuBLAS path only */
    float *h_scores;                                  /* pinned, cuBLAS path */
};

/* The smallest chunk any version uses decides how many partial lists a query
 * can produce, so sizing on that covers v3 (1024) and v4/v5 (256) alike. */
static int kf_max_partials(int N) {
    const int c4 = (N + V4_CHUNK - 1) / V4_CHUNK;
    const int c5 = (N + V5_CHUNK - 1) / V5_CHUNK;
    const int c3 = (N + KF_CHUNK - 1) / KF_CHUNK;
    int m = c4 > c5 ? c4 : c5;
    return m > c3 ? m : c3;
}

extern "C" KfCorpus *kf_corpus_create(const float *X, int N, int d, int *status) {
    cudaError_t e;
    KfCorpus *c = (KfCorpus *)calloc(1, sizeof(KfCorpus));
    if (c == NULL) { if (status) *status = -1; return NULL; }
    c->N = N;
    c->d = d;

    e = cudaMalloc(&c->d_X, (size_t)N * d * sizeof(float));
    if (e == cudaSuccess)
        e = cudaMemcpy(c->d_X, X, (size_t)N * d * sizeof(float), cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        if (c->d_X) cudaFree(c->d_X);
        free(c);
        if (status) *status = (int)e;
        return NULL;
    }
    if (status) *status = 0;
    return c;
}

extern "C" void kf_corpus_destroy(KfCorpus *c) {
    if (c == NULL) return;
    if (c->h_scores) cudaFreeHost(c->h_scores);
    if (c->d_scores) cudaFree(c->d_scores);
    if (c->d_oi) cudaFree(c->d_oi);
    if (c->d_ov) cudaFree(c->d_ov);
    if (c->d_pi) cudaFree(c->d_pi);
    if (c->d_pv) cudaFree(c->d_pv);
    if (c->d_q) cudaFree(c->d_q);
    if (c->d_X) cudaFree(c->d_X);
    free(c);
}

/* Grow a device buffer only when the request exceeds what is already held. */
static cudaError_t kf_reserve(void **buf, int *cap, int want, size_t elem) {
    if (*cap >= want) return cudaSuccess;
    if (*buf) cudaFree(*buf);
    *buf = NULL;
    cudaError_t e = cudaMalloc(buf, (size_t)want * elem);
    *cap = (e == cudaSuccess) ? want : 0;
    return e;
}

extern "C" int kf_corpus_query(KfCorpus *c, int impl, const float *q, int B, int k,
                               float *out_vals, int *out_idx, KfTiming *timing) {
    int status = 0;
    cudaEvent_t ev0 = NULL, ev1 = NULL;
    double t_start = kf_now_ms(), t_h2d0, t_h2d1, t_d2h0, t_d2h1;
    double t_topk0 = 0.0, t_topk1 = 0.0;
    float kernel_ms = 0.0f;
    const int N = c->N, d = c->d;
    const int n_part = kf_max_partials(N);

    if (k < 1 || k > KF_MAX_K) return -1;
    if (impl < KF_IMPL_V3 || impl > KF_IMPL_CUBLAS) return -1;

    KF_CHECK(kf_reserve((void **)&c->d_q,  &c->q_cap,  B * d,          sizeof(float)));
    KF_CHECK(kf_reserve((void **)&c->d_ov, &c->ov_cap, B * k,          sizeof(float)));
    KF_CHECK(kf_reserve((void **)&c->d_oi, &c->oi_cap, B * k,          sizeof(int)));
    KF_CHECK(kf_reserve((void **)&c->d_pv, &c->pv_cap, B * n_part * k, sizeof(float)));
    KF_CHECK(kf_reserve((void **)&c->d_pi, &c->pi_cap, B * n_part * k, sizeof(int)));
    if (impl == KF_IMPL_CUBLAS) {
        const size_t want = (size_t)B * N;
        if (c->score_cap < want) {
            if (c->d_scores) cudaFree(c->d_scores);
            if (c->h_scores) cudaFreeHost(c->h_scores);
            c->d_scores = NULL;
            c->h_scores = NULL;
            c->score_cap = 0;
            KF_CHECK(cudaMalloc(&c->d_scores, want * sizeof(float)));
            KF_CHECK(cudaMallocHost(&c->h_scores, want * sizeof(float)));
            c->score_cap = want;
        }
    }

    KF_CHECK(cudaEventCreate(&ev0));
    KF_CHECK(cudaEventCreate(&ev1));

    /* The only transfer in: the query vectors. At B = 32 that is 48 KB. */
    t_h2d0 = kf_now_ms();
    KF_CHECK(cudaMemcpy(c->d_q, q, (size_t)B * d * sizeof(float), cudaMemcpyHostToDevice));
    KF_CHECK(cudaDeviceSynchronize());
    t_h2d1 = kf_now_ms();

    KF_CHECK(cudaEventRecord(ev0));
    switch (impl) {
        case KF_IMPL_V3:
            status = kf_launch_v3(c->d_q, c->d_X, B, N, d, k, c->d_pv, c->d_pi, c->d_ov, c->d_oi);
            break;
        case KF_IMPL_V4:
            status = kf_launch_v4(c->d_q, c->d_X, B, N, d, k, c->d_pv, c->d_pi, c->d_ov, c->d_oi);
            break;
        case KF_IMPL_V5:
            status = kf_launch_v5(c->d_q, c->d_X, B, N, d, k, c->d_pv, c->d_pi, c->d_ov, c->d_oi);
            break;
        default:
            status = kf_launch_cublas(c->d_q, c->d_X, B, N, d, c->d_scores);
            break;
    }
    if (status != 0) goto cleanup;
    KF_CHECK(cudaEventRecord(ev1));
    KF_CHECK(cudaEventSynchronize(ev1));
    KF_CHECK(cudaEventElapsedTime(&kernel_ms, ev0, ev1));

    t_d2h0 = kf_now_ms();
    if (impl == KF_IMPL_CUBLAS) {
        /* Still the whole score matrix, because the selection is still on the
         * host. That is the point of keeping this row in the comparison. */
        KF_CHECK(cudaMemcpy(c->h_scores, c->d_scores, (size_t)B * N * sizeof(float),
                            cudaMemcpyDeviceToHost));
        t_d2h1 = kf_now_ms();
        t_topk0 = kf_now_ms();
        for (int b = 0; b < B; ++b)
            kf_host_topk(c->h_scores + (size_t)b * N, N, k, out_vals + b * k, out_idx + b * k);
        t_topk1 = kf_now_ms();
    } else {
        KF_CHECK(cudaMemcpy(out_vals, c->d_ov, (size_t)B * k * sizeof(float), cudaMemcpyDeviceToHost));
        KF_CHECK(cudaMemcpy(out_idx, c->d_oi, (size_t)B * k * sizeof(int), cudaMemcpyDeviceToHost));
        t_d2h1 = kf_now_ms();
    }

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
    return status;
}
