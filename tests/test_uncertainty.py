"""Guard paired users, macro weights, exclusion counts, and deterministic resampling."""

import random

import numpy as np
import pytest

from memrecagent import uncertainty as u


def test_identical_models_with_different_user_recalls_have_zero_width_interval():
    ranks = {"a": ["A"], "b": ["Z"], "empty": []}
    result = u.paired_recall_bootstrap(ranks, ranks, {"a": {"A"}, "b": {"B"}, "empty": set()}, k=1)
    assert result["mean_a"] == result["mean_b"] == 0.5
    assert result["delta_a_minus_b"] == 0
    assert result["interval"] == [0, 0]
    assert set(result["bootstrap_deltas"]) == {0}
    assert result["n_users"] == 3 and result["n_evaluated"] == 2
    assert result["n_without_relevant"] == 1


def test_constant_half_advantage_is_preserved_in_every_resample():
    result = u.paired_recall_bootstrap(
        {"u": ["A"], "v": ["C"]},
        {"u": [], "v": []},
        {"u": {"A", "B"}, "v": {"C", "D"}},
        k=1,
        n_resamples=100,
    )
    assert result["delta_a_minus_b"] == 0.5
    assert result["interval"] == [0.5, 0.5]
    assert set(result["bootstrap_deltas"]) == {0.5}


def test_opposite_users_keep_paired_difference_and_use_replacement():
    result = u.paired_recall_bootstrap(
        {"u": ["A"], "v": []},
        {"u": [], "v": ["B"]},
        {"u": {"A"}, "v": {"B"}},
        k=1,
        n_resamples=1000,
    )
    assert result["delta_a_minus_b"] == 0
    # Two draws from differences [1,-1] give only -1,0,1; no replacement would give only0.
    assert set(result["bootstrap_deltas"]) == {-1, 0, 1}
    assert result["interval"] == [-1, 1]


def test_macro_recall_is_not_replaced_by_total_hits_over_total_targets():
    result = u.paired_recall_bootstrap(
        {"u": ["A"], "v": ["B"]},
        {"u": [], "v": ["B"]},
        {"u": {"A"}, "v": {"B", "C", "D", "E"}},
        k=1,
    )
    assert result["mean_a"] == 5 / 8
    assert result["mean_b"] == 1 / 8
    assert result["delta_a_minus_b"] == 1 / 2
    assert result["per_user"] == [
        {"user_id": "u", "recall_a": 1, "recall_b": 0, "difference": 1},
        {"user_id": "v", "recall_a": 0.25, "recall_b": 0.25, "difference": 0},
    ]


def test_reverse_comparison_negates_same_resampled_differences():
    a, b, relevant = {"u": ["A"], "v": []}, {"u": [], "v": ["C"]}, {"u": {"A"}, "v": {"B", "C"}}
    first = u.paired_recall_bootstrap(a, b, relevant, k=1, n_resamples=100)
    second = u.paired_recall_bootstrap(b, a, relevant, k=1, n_resamples=100)
    assert second["delta_a_minus_b"] == -first["delta_a_minus_b"]
    assert second["bootstrap_deltas"] == [-x for x in first["bootstrap_deltas"]]
    assert second["interval"] == pytest.approx([-first["interval"][1], -first["interval"][0]])


def test_mapping_order_does_not_change_sampling_or_global_random_state():
    a, b, rel = {"u": ["A"], "v": []}, {"u": [], "v": []}, {"u": {"A"}, "v": {"B"}}
    py_state = random.getstate()
    np_state = np.random.get_state()
    first = u.paired_recall_bootstrap(a, b, rel, k=1, n_resamples=100)
    second = u.paired_recall_bootstrap(
        dict(reversed(a.items())), b, dict(reversed(rel.items())), k=1, n_resamples=100
    )
    assert first == second
    assert random.getstate() == py_state
    now = np.random.get_state()
    assert now[0] == np_state[0] and np.array_equal(now[1], np_state[1]) and now[2:] == np_state[2:]


@pytest.mark.parametrize(
    "a,b,rel",
    [
        ({"u": []}, {}, {"u": {"A"}}),
        ({"u": []}, {"u": []}, {}),
        ({"u": ["A", "A"]}, {"u": []}, {"u": {"A"}}),
        ({"u": []}, {"u": []}, {"u": set()}),
    ],
)
def test_bad_cohort_or_no_eligible_user_is_rejected(a, b, rel):
    with pytest.raises(ValueError):
        u.paired_recall_bootstrap(a, b, rel, k=1)


@pytest.mark.parametrize(
    "kwargs", [{"n_resamples": 0}, {"confidence": 1}, {"confidence": float("nan")}, {"k": 0}]
)
def test_invalid_bootstrap_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        u.paired_recall_bootstrap({"u": ["A"]}, {"u": []}, {"u": {"A"}}, **kwargs)
