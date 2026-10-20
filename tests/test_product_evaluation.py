"""商品任务的答案、候选和历史必须各自守住时间与分母。"""

import pytest

from memrecagent.amazon import AmazonReview
from memrecagent.product_evaluation import (
    build_window,
    evaluate_window,
    fit_popularity,
    predict_popularity,
)

T = 1640995200000


def event(user, item, rating, timestamp):
    return AmazonReview(user, item, item + "-v", rating, timestamp, "synthetic", 1)


def rows():
    return [
        event("u", "A", 5, T - 2),
        event("u", "B", 1, T - 1),
        event("v", "C", 5, T - 1),
        event("u", "A", 5, T),
        event("u", "D", 1, T),
        event("u", "D", 5, T + 1),
        event("u", "C", 5, T + 2),
        event("cold", "E", 5, T + 2),
        event("low", "F", 1, T + 3),
        event("excluded", "G", 5, T + 10),
    ]


def test_first_new_parent_defines_target_repeats_and_low_only_users_remain_explicit():
    window = build_window(rows(), T, T + 10)
    assert window.users == ("cold", "low", "u")
    assert window.relevant == {"cold": {"E"}, "low": set(), "u": {"C"}}
    assert window.seen["u"] == {"A", "B"}
    assert window.stats["n_window_events"] == 6
    assert window.stats["n_repeat_events"] == 1
    assert window.stats["n_new_pairs"] == 4
    assert window.stats["n_duplicate_new_events"] == 1
    assert window.stats["n_cold_users"] == 2
    target = next(r for r in window.targets if r.user_id == "u" and r.item_id == "D")
    assert target.rating == 1 and target.timestamp_ms == T


def test_popularity_predicts_without_labels_and_filters_all_observations():
    window = build_window(rows(), T, T + 10)
    counts = fit_popularity(window.history, ["A", "B", "C"])
    assert counts == {"A": 1, "B": 0, "C": 1}
    predictions = predict_popularity(counts, window.seen, window.users, k=2)
    assert predictions == {"cold": ["A", "C"], "low": ["A", "C"], "u": ["C"]}
    result = evaluate_window(predictions, window, counts.keys(), k=2)
    assert result["metrics"]["n_evaluated"] == 2
    assert result["metrics"]["n_without_relevant"] == 1
    assert result["metrics"]["mean"] == {
        "precision": 0.25,
        "recall": 0.5,
        "hit_rate": 0.5,
        "ndcg": 0.5,
    }
    assert result["stats"]["n_positive_targets_outside_catalog"] == 1


def test_future_labels_change_answers_but_not_counts_candidates_or_predictions():
    original = rows()
    changed = [
        event(r.user_id, r.item_id, 1 if r.timestamp_ms >= T else r.rating, r.timestamp_ms)
        for r in original
    ]
    first, second = build_window(original, T, T + 10), build_window(changed, T, T + 10)
    a, b = (
        fit_popularity(first.history, ["A", "B", "C"]),
        fit_popularity(second.history, ["A", "B", "C"]),
    )
    assert a == b
    assert predict_popularity(a, first.seen, first.users, 2) == predict_popularity(
        b, second.seen, second.users, 2
    )
    assert second.relevant == {"cold": set(), "low": set(), "u": set()}


def test_empty_candidates_do_not_delete_targets_or_invent_recommendations():
    window = build_window(rows(), T, T + 10)
    predictions = predict_popularity({}, window.seen, window.users, 10)
    assert predictions == {"cold": [], "low": [], "u": []}
    result = evaluate_window(predictions, window, [], 10)
    assert result["metrics"]["mean"]["recall"] == 0
    assert result["stats"]["n_positive_targets_outside_catalog"] == 2
    assert result["stats"]["n_users_without_candidates"] == 3


def test_catalog_cannot_add_future_items_and_candidate_ids_must_be_unique():
    history = build_window(rows(), T, T + 10).history
    for candidates in [["A", "FUTURE"], ["A", "A"]]:
        with pytest.raises(ValueError):
            fit_popularity(history, candidates)


def test_conflicting_same_timestamp_target_is_not_arbitrarily_labelled():
    with pytest.raises(ValueError, match="synthetic:1"):
        build_window([event("u", "A", 5, T), event("u", "A", 1, T)], T, T + 10)


@pytest.mark.parametrize("prediction", [["A"], ["C", "C"], ["FUTURE"], ["C", "D", "E"]])
def test_evaluation_rejects_seen_duplicate_outside_or_oversized_recommendations(prediction):
    window = build_window(rows(), T, T + 10)
    with pytest.raises(ValueError):
        evaluate_window({"cold": [], "low": [], "u": prediction}, window, ["A", "B", "C"], 2)
