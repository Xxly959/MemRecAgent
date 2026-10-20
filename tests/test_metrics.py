"""Hand-derived examples protect ranking and aggregation definitions."""

import pytest

from memrecagent import metrics


def test_lesson_example_distinguishes_precision_and_recall():
    result = metrics.evaluate_ranking(["C", "F", "E"], {"C", "E"}, k=3)
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == 1.0
    assert result.hit_rate == 1.0
    assert result.ndcg == pytest.approx(0.9197207891)


def test_order_changes_ndcg_but_not_hit_counts():
    first = metrics.evaluate_ranking(["E", "F"], {"E"}, k=2)
    second = metrics.evaluate_ranking(["F", "E"], {"E"}, k=2)
    assert first.precision == second.precision == 0.5
    assert first.recall == second.recall == 1.0
    assert first.hit_rate == second.hit_rate == 1.0
    assert first.ndcg == 1.0
    assert second.ndcg == pytest.approx(0.6309297536)


def test_only_top_k_counts():
    result = metrics.evaluate_ranking(["F", "E", "G"], {"E", "G", "H"}, k=2)
    assert result.precision == 0.5
    assert result.recall == pytest.approx(1 / 3)
    assert result.hit_rate == 1.0
    assert result.ndcg == pytest.approx(0.3868528072)


def test_short_list_keeps_requested_k_as_precision_denominator():
    result = metrics.evaluate_ranking(["E"], {"E"}, k=3)
    assert result.precision == pytest.approx(1 / 3)
    assert result.recall == result.ndcg == result.hit_rate == 1.0


def test_empty_recommendations_are_zero_not_skipped():
    result = metrics.evaluate_ranking([], {"E"}, k=3)
    assert result.precision == result.recall == result.hit_rate == result.ndcg == 0.0


def test_no_hits_are_zero():
    result = metrics.evaluate_ranking(["X", "Y"], {"E"}, k=2)
    assert result.precision == result.recall == result.hit_rate == result.ndcg == 0.0


def test_ideal_dcg_is_capped_at_k():
    result = metrics.evaluate_ranking(["C", "E"], {"C", "E", "G"}, k=2)
    assert result.precision == result.ndcg == 1.0
    assert result.recall == pytest.approx(2 / 3)


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_is_invalid(k):
    with pytest.raises(ValueError):
        metrics.evaluate_ranking(["E"], {"E"}, k=k)


def test_duplicate_recommendations_are_invalid():
    with pytest.raises(ValueError):
        metrics.evaluate_ranking(["E", "E"], {"E"}, k=2)


def test_empty_relevance_has_no_single_user_score():
    with pytest.raises(ValueError):
        metrics.evaluate_ranking(["E"], set(), k=2)


def test_macro_mean_weights_users_equally_and_reports_skips():
    result = metrics.evaluate_users(
        {"u1": ["E"], "u2": ["F"], "u3": []},
        {"u1": {"E"}, "u2": {"F", "G", "H"}, "u3": set()},
        k=2,
    )
    assert result.n_users == 3
    assert result.n_evaluated == 2
    assert result.n_without_relevant == 1
    assert result.mean.precision == 0.5
    assert result.mean.recall == pytest.approx(2 / 3)
    assert result.mean.hit_rate == 1.0
    assert result.mean.ndcg == pytest.approx(0.8065735964)


def test_missing_user_is_an_error_not_a_hidden_skip():
    with pytest.raises(ValueError):
        metrics.evaluate_users({"u1": ["E"]}, {"u1": {"E"}, "u2": {"F"}}, k=2)


@pytest.mark.parametrize("recommendations,relevant", [({}, {}), ({"u": []}, {"u": set()})])
def test_no_eligible_users_returns_unavailable_mean(recommendations, relevant):
    result = metrics.evaluate_users(recommendations, relevant, k=2)
    assert result.mean is None
    assert result.n_evaluated == 0
    assert result.n_users == result.n_without_relevant == len(relevant)


def test_zero_recommendations_user_remains_in_macro_denominator():
    result = metrics.evaluate_users({"u1": ["E"], "u2": []}, {"u1": {"E"}, "u2": {"F"}}, k=1)
    assert result.n_evaluated == 2
    assert result.mean.precision == result.mean.recall == 0.5
    assert result.mean.hit_rate == result.mean.ndcg == 0.5
