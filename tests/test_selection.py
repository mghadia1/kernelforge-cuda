"""v3's two-stage selection, tested without a GPU.

v0-v2 hand a plain score matrix to a serial host top-k, so their answer cannot
depend on how the work was split. v3's can: it selects per block over a chunk of
KF_CHUNK documents, then merges those partial lists. A wrong stride, a candidate
lost at a chunk boundary, or a tie resolved by arrival order all produce a
believable wrong answer.

selection_sim.cpp runs that exact decomposition serially in plain C++ against
the same kf_insert rule the kernel uses, so these tests can attack it on any
machine. They do not prove the CUDA correct - shared-memory hazards, missing
syncs, and the dot product itself are invisible here - but they take the
algorithm out of the set of things the T4 run has to discover.
"""

from __future__ import annotations

import numpy as np
import pytest

import reference
import runner

pytestmark = pytest.mark.skipif(
    not runner.sim_available(),
    reason=f"selection simulation unavailable: {runner.sim_error()}",
)

KF_CHUNK = 1024   # must match src/v3_config.h
KF_MAX_K = 8


def check(scores: np.ndarray, k: int):
    exp_vals, exp_idx = reference.topk(scores, k)
    got_vals, got_idx = runner.sim_select(scores, k)
    np.testing.assert_array_equal(got_idx, exp_idx)
    np.testing.assert_allclose(got_vals, exp_vals, rtol=0, atol=0)


# --- the rule itself --------------------------------------------------------

def test_rule_prefers_higher_score():
    assert runner.sim_outranks(0.9, 100, 0.5, 3)
    assert not runner.sim_outranks(0.5, 3, 0.9, 100)


def test_rule_breaks_ties_by_lower_index():
    assert runner.sim_outranks(0.7, 4, 0.7, 9)
    assert not runner.sim_outranks(0.7, 9, 0.7, 4)
    assert not runner.sim_outranks(0.7, 4, 0.7, 4)   # not better than itself


def test_rule_treats_empty_slots_as_losers():
    assert runner.sim_outranks(-1e30, 7, 0.0, -1)    # anything beats an empty slot
    assert not runner.sim_outranks(1e30, -1, 0.0, 3)  # an empty candidate wins nothing


# --- the pipeline -----------------------------------------------------------

@pytest.mark.parametrize("n", [1, 7, 255, 256, 257, KF_CHUNK - 1, KF_CHUNK,
                               KF_CHUNK + 1, 3 * KF_CHUNK, 5000])
def test_matches_reference_across_chunk_boundaries(n):
    """N on, either side of, and far from a chunk boundary."""
    rng = np.random.default_rng(n)
    scores = rng.standard_normal((3, n), dtype=np.float32)
    check(scores, k=min(5, n))


@pytest.mark.parametrize("k", range(1, KF_MAX_K + 1))
def test_every_supported_k(k):
    rng = np.random.default_rng(k)
    check(rng.standard_normal((2, 4000), dtype=np.float32), k)


def test_k_outside_the_supported_range_is_rejected():
    scores = np.zeros((1, 100), dtype=np.float32)
    for bad in (0, -1, KF_MAX_K + 1, 64):
        with pytest.raises(ValueError):
            runner.sim_select(scores, bad)


def test_all_scores_identical_returns_the_lowest_indices():
    """The worst case for a distributed selection: every tie-break matters, and
    the answer is only stable because the rule compares indices too."""
    scores = np.full((2, 5000), 0.25, dtype=np.float32)
    vals, idx = runner.sim_select(scores, 5)
    assert idx.tolist() == [[0, 1, 2, 3, 4]] * 2
    assert np.all(vals == 0.25)


def test_heavy_ties_still_match_the_reference():
    """Scores quantized to 8 levels over 6 chunks, so ties cross block
    boundaries and the merge stage has to resolve them the same way argsort
    does."""
    rng = np.random.default_rng(0)
    scores = (rng.integers(0, 8, size=(4, 6 * KF_CHUNK)) / 8.0).astype(np.float32)
    check(scores, k=8)


def test_winners_hiding_in_the_last_partial_chunk():
    """The tail chunk is shorter than KF_CHUNK and its threads run fewer strides.
    If the tail were mishandled these top scores would vanish."""
    n = 3 * KF_CHUNK + 17
    scores = np.zeros((1, n), dtype=np.float32)
    scores[0, -3:] = [0.91, 0.93, 0.92]
    vals, idx = runner.sim_select(scores, 5)
    assert idx[0, :3].tolist() == [n - 2, n - 1, n - 3]
    np.testing.assert_allclose(vals[0, :3], [0.93, 0.92, 0.91])


def test_winners_spread_one_per_chunk():
    """One high score in each chunk: every partial list must contribute, so this
    fails if the merge only ever reads the first block's candidates."""
    chunks = 6
    n = chunks * KF_CHUNK
    scores = np.zeros((1, n), dtype=np.float32)
    planted = [c * KF_CHUNK + 11 * c for c in range(chunks)]
    for rank, pos in enumerate(planted):
        scores[0, pos] = 0.9 - 0.01 * rank
    vals, idx = runner.sim_select(scores, 5)
    assert idx[0].tolist() == planted[:5]


def test_matches_the_real_embedding_size():
    """The actual retrieval case: PaperTrail's 2039 chunks against real-shaped
    normalized vectors, not a synthetic score matrix."""
    q, x = reference.make_data(2039, 8, seed=42)
    check(reference.cosine_scores(q, x), k=5)
