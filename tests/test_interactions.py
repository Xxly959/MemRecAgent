"""Check interaction conversion against small, hand-derived matrices."""

import numpy as np
import pytest


def test_builds_matrix_in_first_seen_id_order() -> None:
    from memrecagent.interactions import build_user_item_matrix

    records = [("小周", "B", 5.0), ("小林", "A", 5.0), ("小周", "A", 4.0)]
    ratings, observed, users, items = build_user_item_matrix(records)

    assert users == {"小周": 0, "小林": 1}
    assert items == {"B": 0, "A": 1}
    np.testing.assert_array_equal(ratings, [[5.0, 4.0], [0.0, 5.0]])
    np.testing.assert_array_equal(observed, [[True, True], [False, True]])
    assert ratings.dtype == np.float32
    assert observed.dtype == np.bool_


def test_low_rating_is_still_observed() -> None:
    from memrecagent.interactions import build_user_item_matrix

    ratings, observed, _, _ = build_user_item_matrix([("u", "i", 1.0)])

    assert ratings[0, 0] == 1.0
    assert observed[0, 0]


def test_empty_input_returns_empty_matrices_and_mappings() -> None:
    from memrecagent.interactions import build_user_item_matrix

    ratings, observed, users, items = build_user_item_matrix([])

    assert ratings.shape == (0, 0)
    assert observed.shape == (0, 0)
    assert ratings.dtype == np.float32
    assert observed.dtype == np.bool_
    assert users == {}
    assert items == {}


@pytest.mark.parametrize("second_rating", [5.0, 1.0])
def test_duplicate_pair_is_rejected(second_rating: float) -> None:
    from memrecagent.interactions import build_user_item_matrix

    with pytest.raises(ValueError, match="Duplicate"):
        build_user_item_matrix([("u", "i", 5.0), ("u", "i", second_rating)])


def test_input_records_are_not_modified() -> None:
    from memrecagent.interactions import build_user_item_matrix

    records = [("z", "B", 5.0), ("a", "A", 1.0)]
    original = records.copy()

    build_user_item_matrix(records)

    assert records == original
