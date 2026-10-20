"""Check filtering, tie-breaking, and Top-K against literal expected lists."""

import pytest


@pytest.mark.parametrize(
    ("seen_items", "k", "expected"),
    [
        (set(), 2, ["C", "D"]),
        ({"C", "D"}, 2, ["F", "E"]),
        ({"D", "F"}, 2, ["C", "E"]),
        ({"C", "D", "F"}, 2, ["E"]),
        ({"C", "D", "E", "F"}, 2, []),
        (set(), 0, []),
    ],
)
def test_filters_seen_items_before_ranking_and_limits_results(
    seen_items: set[str], k: int, expected: list[str]
) -> None:
    from memrecagent.popularity import recommend_popular

    popularity = {"C": 3, "D": 2, "E": 1, "F": 2}

    assert recommend_popular(popularity, seen_items, k) == expected


def test_ties_use_item_id_order_instead_of_dictionary_order() -> None:
    from memrecagent.popularity import recommend_popular

    assert recommend_popular({"F": 2, "D": 2}, set(), 2) == ["D", "F"]


def test_empty_catalog_returns_empty_list() -> None:
    from memrecagent.popularity import recommend_popular

    assert recommend_popular({}, {"C"}, 2) == []


def test_negative_k_is_rejected() -> None:
    from memrecagent.popularity import recommend_popular

    with pytest.raises(ValueError, match="non-negative"):
        recommend_popular({"C": 3}, set(), -1)


def test_scores_and_seen_items_are_not_modified() -> None:
    from memrecagent.popularity import recommend_popular

    popularity = {"F": 2, "C": 3, "D": 2}
    seen_items = {"C"}

    recommend_popular(popularity, seen_items, 2)

    assert list(popularity.items()) == [("F", 2), ("C", 3), ("D", 2)]
    assert seen_items == {"C"}


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [(4.0, {"C": 1, "D": 1, "E": 2}), (5.0, {"C": 1, "D": 0, "E": 2})],
)
def test_counts_positive_users_and_keeps_zero_count_items(
    threshold: float, expected: dict[str, int]
) -> None:
    from memrecagent.popularity import count_positive_ratings

    records = [
        ("u1", "C", 5.0),
        ("u1", "D", 4.0),
        ("u1", "E", 5.0),
        ("u2", "C", 2.0),
        ("u2", "D", 1.0),
        ("u2", "E", 5.0),
    ]

    assert count_positive_ratings(records, threshold=threshold) == expected


def test_counting_empty_records_returns_empty_dictionary() -> None:
    from memrecagent.popularity import count_positive_ratings

    assert count_positive_ratings([]) == {}


@pytest.mark.parametrize("second_rating", [5.0, 1.0])
def test_counting_rejects_duplicate_user_item_pair(second_rating: float) -> None:
    from memrecagent.popularity import count_positive_ratings

    with pytest.raises(ValueError, match="Duplicate"):
        count_positive_ratings([("u1", "C", 5.0), ("u1", "C", second_rating)])


def test_counting_does_not_modify_records() -> None:
    from memrecagent.popularity import count_positive_ratings

    records = [("u2", "E", 5.0), ("u1", "C", 2.0)]
    original = records.copy()

    count_positive_ratings(records)

    assert records == original


def test_training_records_feed_popularity_ranking() -> None:
    from memrecagent.popularity import count_positive_ratings, recommend_popular

    records = [
        ("u1", "C", 5.0),
        ("u1", "D", 4.0),
        ("u1", "E", 5.0),
        ("u2", "C", 2.0),
        ("u2", "D", 1.0),
        ("u2", "E", 5.0),
        ("target", "E", 1.0),
    ]

    popularity = count_positive_ratings(records)

    assert recommend_popular(popularity, {"E"}, k=2) == ["C", "D"]
