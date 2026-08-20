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
V4_CHUNK = 256    # must match src/v4_config.h
KF_MAX_K = 8


def _config(header: str, name: str) -> int:
    """Read a #define out of a config header, so the tests cannot silently
    drift from the constants the kernels actually compile with."""
    import pathlib as _p
    import re as _re
    text = (_p.Path(__file__).resolve().parent.parent / "src" / header).read_text()
    m = _re.search(rf"#define\s+{name}\s+(\d+)", text)
    assert m, f"{name} not found in {header}"
    return int(m.group(1))

# Both selections are exercised by every pipeline test below. v4's is the newer
# and stranger one: one warp per query, k rounds of a masked lexicographic
# max-reduction over lane registers, no shared scratch. Its distinctive failure
# mode is a lane that fails to clear its used-bit, which returns the same
# document k times — cheap to catch here, expensive to discover on a GPU.
VERSIONS = ["v3", "v4"]


def check(scores: np.ndarray, k: int, version: str = "v3"):
    exp_vals, exp_idx = reference.topk(scores, k)
    got_vals, got_idx = runner.sim_select(scores, k, version)
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

@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("n", [1, 7, 31, 32, 33, 255, 256, 257, KF_CHUNK - 1,
                               KF_CHUNK, KF_CHUNK + 1, 3 * KF_CHUNK, 5000])
def test_matches_reference_across_chunk_boundaries(n, version):
    """N on, either side of, and far from a chunk boundary. The small sizes also
    cover v4's lane geometry: fewer documents than one warp, exactly one warp,
    and one more than a warp."""
    rng = np.random.default_rng(n)
    scores = rng.standard_normal((3, n), dtype=np.float32)
    check(scores, k=min(5, n), version=version)


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("k", range(1, KF_MAX_K + 1))
def test_every_supported_k(k, version):
    rng = np.random.default_rng(k)
    check(rng.standard_normal((2, 4000), dtype=np.float32), k, version)


def test_k_outside_the_supported_range_is_rejected():
    scores = np.zeros((1, 100), dtype=np.float32)
    for bad in (0, -1, KF_MAX_K + 1, 64):
        with pytest.raises(ValueError):
            runner.sim_select(scores, bad)


@pytest.mark.parametrize("version", VERSIONS)
def test_no_document_is_selected_twice(version):
    """v4's used-bit mask is what stops a lane re-offering the document it just
    won. Distinct scores make any repeat obvious."""
    rng = np.random.default_rng(99)
    scores = rng.standard_normal((4, 4 * V4_CHUNK + 5), dtype=np.float32)
    _, idx = runner.sim_select(scores, 8, version)
    for row in idx:
        assert len(set(row.tolist())) == len(row)


@pytest.mark.parametrize("version", VERSIONS)
def test_all_winners_inside_one_lane(version):
    """Every top score sits in entries that a single lane owns (indices congruent
    mod 32 within one chunk), so one lane must give up k documents in k rounds.
    A used-bit that never clears returns the same index k times."""
    n = 2 * V4_CHUNK
    scores = np.zeros((1, n), dtype=np.float32)
    planted = [7 + 32 * e for e in range(5)]        # lane 7, entries 0..4
    for rank, pos in enumerate(planted):
        scores[0, pos] = 0.9 - 0.01 * rank
    vals, idx = runner.sim_select(scores, 5, version)
    assert idx[0].tolist() == planted


def test_all_scores_identical_returns_the_lowest_indices():
    """The worst case for a distributed selection: every tie-break matters, and
    the answer is only stable because the rule compares indices too."""
    scores = np.full((2, 5000), 0.25, dtype=np.float32)
    vals, idx = runner.sim_select(scores, 5)
    assert idx.tolist() == [[0, 1, 2, 3, 4]] * 2
    assert np.all(vals == 0.25)


@pytest.mark.parametrize("version", VERSIONS)
def test_heavy_ties_still_match_the_reference(version):
    """Scores quantized to 8 levels over 6 chunks, so ties cross block
    boundaries and the merge stage has to resolve them the same way argsort
    does."""
    rng = np.random.default_rng(0)
    scores = (rng.integers(0, 8, size=(4, 6 * KF_CHUNK)) / 8.0).astype(np.float32)
    check(scores, k=8, version=version)


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


@pytest.mark.parametrize("version", VERSIONS)
def test_matches_the_real_embedding_size(version):
    """The actual retrieval case: PaperTrail's 2039 chunks against real-shaped
    normalized vectors, not a synthetic score matrix."""
    q, x = reference.make_data(2039, 8, seed=42)
    check(reference.cosine_scores(q, x), k=5, version=version)


@pytest.mark.parametrize("version", VERSIONS)
def test_heavy_ties_inside_one_v4_chunk(version):
    """Ties dense enough that several land in the same lane's entries, where v4
    resolves them across rounds rather than in one pass."""
    rng = np.random.default_rng(5)
    scores = (rng.integers(0, 3, size=(3, 3 * V4_CHUNK)) / 4.0).astype(np.float32)
    check(scores, k=8, version=version)


# --- v5 rides on v4's selection ---------------------------------------------

def test_v5_selection_geometry_matches_v4():
    """v5 changes only the scoring loop; it calls the same kf_warp_select over
    the same chunk. That is what lets the v4 simulation above stand as coverage
    for v5's selection as well. If someone retunes V5_CHUNK, this fails loudly
    rather than leaving a coverage claim quietly false."""
    assert _config("v5_config.h", "V5_CHUNK") == _config("v4_config.h", "V4_CHUNK")
    assert _config("v5_config.h", "V5_QT") == _config("v4_config.h", "V4_QT")
    assert _config("v5_config.h", "V5_PER_LANE".replace("V5_PER_LANE", "V5_CHUNK")) // 32 == V4_CHUNK // 32


def test_config_constants_match_the_test_file():
    """The chunk sizes hardcoded at the top of this file are the ones the tests
    reason about; keep them tied to the headers."""
    assert _config("v3_config.h", "KF_CHUNK") == KF_CHUNK
    assert _config("v4_config.h", "V4_CHUNK") == V4_CHUNK
    assert _config("v3_config.h", "KF_MAX_K") == KF_MAX_K
