"""Protect history-only routing and preserve chosen ranking order."""

import pytest

from memrecagent import fallback as f


def test_history_user_keeps_bpr_while_new_user_gets_popularity():
    assert f.recommend_with_fallback("小林", {"小林"}, ["D", "E", "C"], ["C", "D", "E"], 2) == [
        "D",
        "E",
    ]
    assert f.recommend_with_fallback("小周", {"小林"}, ["D", "E", "C"], ["C", "D", "E"], 2) == [
        "C",
        "D",
    ]


def test_history_user_with_empty_bpr_list_is_not_reclassified_as_cold():
    assert f.recommend_with_fallback("u", {"u"}, [], ["A", "B"], 2) == []


def test_empty_cold_list_does_not_invent_items_from_other_ranker():
    assert f.recommend_with_fallback("new", {"u"}, ["A"], [], 2) == []


def test_short_lists_are_copied_without_filling_or_mutating_inputs():
    bpr, popular = ["D"], ["C"]
    result = f.recommend_with_fallback("new", {"u"}, bpr, popular, 5)
    assert result == ["C"]
    result.append("E")
    assert popular == ["C"] and bpr == ["D"]


@pytest.mark.parametrize("k", [0, -1])
def test_nonpositive_k_is_rejected(k):
    with pytest.raises(ValueError):
        f.recommend_with_fallback("u", {"u"}, ["A"], ["B"], k)


@pytest.mark.parametrize("user", ["u", "new"])
def test_duplicate_items_in_selected_list_are_rejected(user):
    with pytest.raises(ValueError):
        f.recommend_with_fallback(user, {"u"}, ["A", "A"], ["B", "B"], 2)
