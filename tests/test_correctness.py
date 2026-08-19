"""Correctness of every kernel version against the NumPy reference.

The GPU tests skip when there is no CUDA device or the library has not been
built, so this file is meaningful in two places: on a Colab/Kaggle T4 it is the
gate that says the kernels are right, and on CPU-only CI the reference tests
below still run and still fail if the ground truth itself breaks.
"""

from __future__ import annotations

import numpy as np
import pytest

import reference
import runner

# Sizes chosen to hit the awkward cases, not just the round ones:
#   2039  - PaperTrail's real chunk count, and not a multiple of any tile size
#   1     - single document, shorter than one warp's worth of work
#   1024  - exactly v3's CHUNK, so the tail path is empty
#   50000 - several partial-top-k blocks per query
SIZES = [(2039, 4), (1, 3), (1024, 1), (50000, 8)]
TOL = 1e-4

gpu_required = pytest.mark.skipif(
    not runner.available(),
    reason=f"no CUDA device or library: {runner.load_error() or 'no device'}",
)


# --- reference tests: these run everywhere, including CPU-only CI ------------

def test_make_data_is_normalized_float32():
    q, x = reference.make_data(64, 4, seed=1)
    assert x.dtype == np.float32 and q.dtype == np.float32
    assert x.flags["C_CONTIGUOUS"] and q.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-5)


def test_reference_matches_a_plain_python_loop():
    q, x = reference.make_data(37, 3, d=16, seed=2)
    vals, idx = reference.reference_topk(q, x, k=5)
    for b in range(q.shape[0]):
        scores = [float(np.dot(q[b], x[n])) for n in range(x.shape[0])]
        expected = sorted(range(len(scores)), key=lambda n: (-scores[n], n))[:5]
        assert list(idx[b]) == expected
        np.testing.assert_allclose(vals[b], [scores[n] for n in expected], atol=1e-6)


def test_reference_breaks_ties_by_lower_index():
    scores = np.array([[0.5, 0.9, 0.9, 0.1]], dtype=np.float32)
    vals, idx = reference.topk(scores, k=3)
    assert list(idx[0]) == [1, 2, 0]
    np.testing.assert_allclose(vals[0], [0.9, 0.9, 0.5])


# --- GPU tests: the actual gate, skipped without a device -------------------

@gpu_required
@pytest.mark.parametrize("version", runner.ALL_IMPLS)
@pytest.mark.parametrize("n,b", SIZES)
def test_kernel_matches_reference(version, n, b):
    k = min(5, n)
    q, x = reference.make_data(n, b, seed=n + b)
    exp_vals, exp_idx = reference.reference_topk(q, x, k)
    got_vals, got_idx, _ = runner.run(version, q, x, k)

    assert np.max(np.abs(got_vals - exp_vals)) < TOL
    # Random float scores do not tie, so the indices must match exactly.
    assert np.array_equal(got_idx, exp_idx)


@gpu_required
@pytest.mark.parametrize("version", runner.ALL_IMPLS)
def test_largest_supported_k(version):
    """k = 8 is the widest list the device-side selection reserves room for."""
    n, b, k = 64, 2, 8
    q, x = reference.make_data(n, b, seed=7)
    exp_vals, _ = reference.reference_topk(q, x, k)
    got_vals, _, _ = runner.run(version, q, x, k)
    np.testing.assert_allclose(got_vals, exp_vals, atol=TOL)


@gpu_required
def test_timing_is_populated():
    q, x = reference.make_data(4096, 2, seed=11)
    _, _, t = runner.run("v3_topk", q, x, 5)
    assert t.kernel_ms > 0.0
    assert t.total_ms >= t.kernel_ms
    # v3 selects on the GPU, so it must not be paying for a host top-k.
    assert t.host_topk_ms == 0.0


@gpu_required
def test_unknown_version_is_rejected():
    q, x = reference.make_data(16, 1, seed=3)
    with pytest.raises(ValueError):
        runner.run("v9_imaginary", q, x, 5)
