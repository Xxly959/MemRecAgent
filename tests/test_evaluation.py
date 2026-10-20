"""Integration fixtures guard shared candidates and history-only fitting."""

import pytest

from memrecagent import evaluation
from memrecagent.movielens import Rating


def history():
    return [
        Rating("u1", "A", 5, 1),
        Rating("u1", "C", 5, 2),
        Rating("u2", "A", 5, 3),
        Rating("u2", "B", 5, 4),
        Rating("target", "A", 5, 5),
    ]


def test_toy_ranking_and_common_string_id_tie():
    result = evaluation.evaluate_holdout(history(), [Rating("target", "C", 5, 10)], k=2)
    for model in ["popularity", "itemcf"]:
        assert result["predictions"][model]["target"] == ["B", "C"]
        assert result["metrics"][model]["mean"]["recall"] == 1.0
        assert result["metrics"][model]["mean"]["precision"] == 0.5
        assert result["metrics"][model]["mean"]["ndcg"] == pytest.approx(0.6309297536)


def test_low_ratings_are_filtered_and_zero_scores_still_rank():
    train = history() + [Rating("target", "B", 1, 6)]
    result = evaluation.evaluate_holdout(train, [Rating("target", "C", 5, 10)], k=2)
    assert result["predictions"]["popularity"]["target"] == ["C"]
    assert result["predictions"]["itemcf"]["target"] == ["C"]


def test_cold_user_and_out_of_catalog_positive_remain_in_denominator():
    result = evaluation.evaluate_holdout(history(), [Rating("new", "Z", 5, 10)], k=2)
    assert result["stats"]["n_cold_users"] == 1
    assert result["stats"]["n_positive_targets_outside_catalog"] == 1
    assert result["stats"]["n_catalog_items"] == 3
    assert result["relevant"]["new"] == ["Z"]
    assert result["predictions"]["itemcf"]["new"] == ["A", "B"]
    assert result["metrics"]["itemcf"]["n_evaluated"] == 1
    assert result["metrics"]["itemcf"]["mean"]["recall"] == 0.0


def test_no_positive_history_is_reported_not_dropped():
    train = history() + [Rating("low", "A", 1, 6)]
    result = evaluation.evaluate_holdout(train, [Rating("low", "B", 5, 10)], k=2)
    assert result["stats"]["n_users_without_positive_history"] == 1
    assert result["predictions"]["itemcf"]["low"] == ["B", "C"]


def test_all_zero_catalog_and_no_test_positive_have_unavailable_mean():
    result = evaluation.evaluate_holdout([Rating("u", "A", 1, 1)], [Rating("u", "B", 2, 10)], k=2)
    for model in ["popularity", "itemcf"]:
        assert result["metrics"][model]["n_without_relevant"] == 1
        assert result["metrics"][model]["mean"] is None


def test_input_order_does_not_change_output():
    future = [Rating("target", "C", 5, 10), Rating("u1", "B", 5, 11)]
    assert evaluation.evaluate_holdout(history(), future) == evaluation.evaluate_holdout(
        list(reversed(history())), list(reversed(future))
    )


def test_future_labels_do_not_change_predictions():
    first = evaluation.evaluate_holdout(history(), [Rating("target", "C", 5, 10)])
    second = evaluation.evaluate_holdout(history(), [Rating("target", "C", 1, 10)])
    assert first["predictions"] == second["predictions"]
    assert first["metrics"] != second["metrics"]


def test_interleaved_time_is_rejected():
    with pytest.raises(ValueError, match="precede"):
        evaluation.evaluate_holdout(history(), [Rating("target", "C", 5, 4)])


def test_repeated_pair_across_history_and_holdout_is_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        evaluation.evaluate_holdout(history(), [Rating("target", "A", 5, 10)])


@pytest.mark.parametrize("threshold,k", [(6, 10), (0, 10), (4, 0)])
def test_invalid_configuration_is_rejected(threshold, k):
    with pytest.raises(ValueError):
        evaluation.evaluate_holdout(history(), [], threshold=threshold, k=k)


def test_empty_fitting_history_is_rejected():
    with pytest.raises(ValueError):
        evaluation.evaluate_holdout([], [Rating("u", "A", 5, 10)])
