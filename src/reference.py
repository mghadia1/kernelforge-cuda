"""NumPy ground truth for the retrieval hot path.

The kernels in this repo are only interesting if they agree with a definition
that is obviously correct, so this file keeps the definition as plain as
possible: a dense matmul and a stable argsort. Everything else in the repo is
measured against it.

Tie-break rule: when two documents score exactly the same, the lower index
wins. ``np.argsort(..., kind="stable")`` gives that for free, and the CUDA
kernels implement the same rule, which is what lets the tests compare returned
indices instead of only scores.
"""

from __future__ import annotations

import numpy as np

DIM = 384  # all-MiniLM-L6-v2, the embedding model PaperTrail uses


def l2_normalize(a: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization; zero rows are left as zeros."""
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (a / norms).astype(np.float32, copy=False)


def make_data(n: int, b: int, d: int = DIM, seed: int = 0):
    """Synthetic normalized corpus and queries, float32 and C-contiguous.

    Synthetic on purpose: this project measures kernel time, not retrieval
    quality, and the timing does not depend on what the vectors mean. Real
    PaperTrail embeddings are used only for the N = 2039 correctness case.
    """
    rng = np.random.default_rng(seed)
    x = l2_normalize(rng.standard_normal((n, d), dtype=np.float32))
    q = l2_normalize(rng.standard_normal((b, d), dtype=np.float32))
    return np.ascontiguousarray(q), np.ascontiguousarray(x)


def cosine_scores(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """(B, N) similarity matrix. Inputs are normalized, so this is q @ X.T."""
    return q.astype(np.float32) @ x.astype(np.float32).T


def topk(scores: np.ndarray, k: int):
    """Top-k per row, descending, ties broken by the lower column index."""
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    vals = np.take_along_axis(scores, order, axis=1)
    return vals.astype(np.float32), order.astype(np.int32)


def reference_topk(q: np.ndarray, x: np.ndarray, k: int):
    """The full CPU reference: score everything, then select."""
    return topk(cosine_scores(q, x), k)
