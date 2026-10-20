"""Verify legal pools, deterministic sampling, and explicit skip counts."""

import random

import pytest

from memrecagent import sampling


def test_pool_excludes_all_own_observations_but_not_other_users_low_ratings():
    sampler = sampling.BPRSampler(
        [
            ("小林", "A", 5.0),
            ("小林", "B", 1.0),
            ("其他", "C", 4.0),
            ("其他", "D", 2.0),
            ("其他", "E", 5.0),
        ]
    )
    assert sampler.catalog == {"A", "B", "C", "D", "E"}
    assert sampler.negative_pools["小林"] == ("C", "D", "E")
    assert sampler.negative_pools["其他"] == ("A", "B")
    assert sampler.positive_pairs == (("其他", "C"), ("其他", "E"), ("小林", "A"))
    result = sampler.sample_epoch(random.Random(42))
    assert len(result.triples) == result.n_positive_pairs == 3
    for user, positive, negative in result.triples:
        assert (user, positive) in sampler.positive_pairs
        assert negative in sampler.catalog
        assert negative not in sampler.observed[user]
    assert "F" not in sampler.catalog


def test_full_catalog_user_skips_positive_pairs_and_no_positive_user_is_separate():
    sampler = sampling.BPRSampler(
        [
            ("full", "A", 5.0),
            ("full", "B", 4.0),
            ("full", "C", 1.0),
            ("normal", "A", 5.0),
            ("low", "B", 2.0),
        ]
    )
    result = sampler.sample_epoch(random.Random(42))
    assert result.n_positive_pairs == 3
    assert result.n_skipped_pairs_no_candidates == 2
    assert result.n_positive_users_without_candidates == 1
    assert result.n_users_without_positives == 1
    assert len(result.triples) == 1
    assert result.triples[0][:2] == ("normal", "A")
    assert result.triples[0][2] in {"B", "C"}


def test_one_candidate_can_be_reused_for_different_positive_pairs():
    sampler = sampling.BPRSampler(
        [
            ("u", "A", 5.0),
            ("u", "B", 4.0),
            ("v", "C", 1.0),
        ]
    )
    result = sampler.sample_epoch(random.Random(42))
    assert result.triples == [("u", "A", "C"), ("u", "B", "C")]


def test_canonical_order_replays_same_rng_stream_across_epochs():
    rows = [("u", "A", 5.0), ("u", "B", 5.0), ("v", "C", 4.0), ("v", "D", 1.0)]
    first = sampling.BPRSampler(rows)
    second = sampling.BPRSampler(list(reversed(rows)))
    a, b = random.Random(42), random.Random(42)
    assert first.sample_epoch(a) == second.sample_epoch(b)
    assert first.sample_epoch(a) == second.sample_epoch(b)


def test_sampler_uses_caller_rng_without_changing_global_random_state():
    sampler = sampling.BPRSampler([("u", "A", 5.0), ("v", "B", 4.0)])
    before = random.getstate()
    sampler.sample_epoch(random.Random(42))
    assert random.getstate() == before


def test_threshold_boundary_and_no_positive_catalog_retention():
    rows = [("u", "A", 4.0), ("v", "B", 5.0), ("w", "C", 1.0)]
    assert sampling.BPRSampler(rows).positive_pairs == (("u", "A"), ("v", "B"))
    sampler = sampling.BPRSampler(rows, threshold=5.0)
    assert sampler.positive_pairs == (("v", "B"),)
    assert sampler.catalog == {"A", "B", "C"}


def test_empty_history_has_no_triples_and_zero_counts():
    result = sampling.BPRSampler([]).sample_epoch(random.Random(42))
    assert result.triples == []
    assert result.n_positive_pairs == result.n_skipped_pairs_no_candidates == 0
    assert result.n_users_without_positives == result.n_positive_users_without_candidates == 0


@pytest.mark.parametrize("rating", [5.0, 1.0])
def test_duplicate_pairs_rejected_without_silent_overwrite(rating):
    with pytest.raises(ValueError, match="Duplicate"):
        sampling.BPRSampler([("u", "A", 5.0), ("u", "A", rating)])


@pytest.mark.parametrize("threshold", [0.0, 6.0, float("nan")])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError):
        sampling.BPRSampler([], threshold=threshold)


@pytest.mark.parametrize("rating", [0.0, 6.0, float("nan")])
def test_invalid_rating_is_rejected(rating):
    with pytest.raises(ValueError):
        sampling.BPRSampler([("u", "A", rating)])
