"""Check the learner's item cooccurrence expression against hand-counted data."""

import numpy as np
import pytest


def test_item_cooccurrence_matches_shared_positive_user_counts() -> None:
    from memrecagent.itemcf import item_cooccurrence

    x = np.array([[1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    result = item_cooccurrence(x)

    np.testing.assert_array_equal(result, [[4, 1, 2], [1, 1, 0], [2, 0, 2]])
    assert result.dtype == np.float32


def test_item_with_no_positive_users_has_zero_cooccurrence() -> None:
    from memrecagent.itemcf import item_cooccurrence

    x = np.array([[1, 0, 1], [0, 0, 1]], dtype=np.float32)

    np.testing.assert_array_equal(item_cooccurrence(x), [[1, 0, 1], [0, 0, 0], [1, 0, 2]])


def test_no_users_retains_the_item_by_item_shape() -> None:
    from memrecagent.itemcf import item_cooccurrence

    x = np.zeros((0, 3), dtype=np.float32)

    np.testing.assert_array_equal(item_cooccurrence(x), np.zeros((3, 3), dtype=np.float32))


@pytest.fixture
def teaching_similarity() -> np.ndarray:
    return np.array([[1, 0.5, 0.70710678], [0.5, 1, 0], [0.70710678, 0, 1]], dtype=np.float32)


def test_cosine_similarity_matches_hand_calculation(teaching_similarity: np.ndarray) -> None:
    from memrecagent.itemcf import cosine_item_similarity

    x = np.array([[1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    result = cosine_item_similarity(x)

    np.testing.assert_allclose(result, teaching_similarity, rtol=1e-6, atol=1e-7)
    assert result.dtype == np.float32


def test_zero_norm_item_produces_finite_zero_similarities() -> None:
    from memrecagent.itemcf import cosine_item_similarity

    x = np.array([[1, 0], [0, 0]], dtype=np.float32)
    with np.errstate(divide="raise", invalid="raise"):
        result = cosine_item_similarity(x)

    np.testing.assert_array_equal(result, [[1, 0], [0, 0]])


def test_cosine_with_no_users_returns_zero_similarities() -> None:
    from memrecagent.itemcf import cosine_item_similarity

    result = cosine_item_similarity(np.zeros((0, 3), dtype=np.float32))

    np.testing.assert_array_equal(result, np.zeros((3, 3), dtype=np.float32))


@pytest.mark.parametrize("x", [np.array([1, 0]), np.array([[5, 0]])])
def test_cosine_rejects_invalid_binary_matrix(x: np.ndarray) -> None:
    from memrecagent.itemcf import cosine_item_similarity

    with pytest.raises(ValueError):
        cosine_item_similarity(x)


def test_scores_add_support_from_multiple_history_items() -> None:
    from memrecagent.itemcf import score_items

    similarity = np.array([[1, 0.5, 0.7], [0.5, 1, 0.2], [0.7, 0.2, 1]], dtype=np.float32)
    history = np.array([1, 0, 1], dtype=np.float32)

    np.testing.assert_allclose(score_items(similarity, history), [1.7, 0.7, 1.7], rtol=1e-6)


@pytest.mark.parametrize(
    ("similarity", "history"),
    [(np.ones((2, 3)), np.ones(3)), (np.eye(3), np.ones((1, 3))), (np.eye(3), np.ones(2))],
)
def test_scoring_rejects_mismatched_shapes(similarity: np.ndarray, history: np.ndarray) -> None:
    from memrecagent.itemcf import score_items

    with pytest.raises(ValueError):
        score_items(similarity, history)


def test_recommendation_orders_unseen_items_by_history_support(
    teaching_similarity: np.ndarray,
) -> None:
    from memrecagent.itemcf import recommend_itemcf

    result = recommend_itemcf(
        teaching_similarity,
        np.array([1, 0, 0], dtype=np.float32),
        np.array([True, False, False]),
        k=2,
    )

    assert result == [2, 1]


def test_low_rated_observed_item_is_filtered(teaching_similarity: np.ndarray) -> None:
    from memrecagent.itemcf import recommend_itemcf

    result = recommend_itemcf(
        teaching_similarity,
        np.array([1, 0, 0], dtype=np.float32),
        np.array([True, False, True]),
        k=2,
    )

    assert result == [1]


def test_no_positive_history_returns_no_personalized_results(
    teaching_similarity: np.ndarray,
) -> None:
    from memrecagent.itemcf import recommend_itemcf

    assert recommend_itemcf(teaching_similarity, np.zeros(3), np.zeros(3, dtype=bool), 2) == []


def test_equal_scores_use_column_index_order() -> None:
    from memrecagent.itemcf import recommend_itemcf

    similarity = np.array([[1, 0.5, 0.5], [0.5, 1, 0], [0.5, 0, 1]], dtype=np.float32)

    assert recommend_itemcf(similarity, np.array([1, 0, 0]), np.array([True, False, False]), 2) == [
        1,
        2,
    ]


def test_zero_support_candidates_are_not_filled_in(teaching_similarity: np.ndarray) -> None:
    from memrecagent.itemcf import recommend_itemcf

    result = recommend_itemcf(
        teaching_similarity, np.array([0, 1, 0]), np.array([False, True, False]), 2
    )

    assert result == [0]


def test_empty_catalog_returns_no_results() -> None:
    from memrecagent.itemcf import recommend_itemcf

    assert recommend_itemcf(np.zeros((0, 0)), np.zeros(0), np.zeros(0, dtype=bool), 2) == []


def test_negative_k_is_rejected(teaching_similarity: np.ndarray) -> None:
    from memrecagent.itemcf import recommend_itemcf

    with pytest.raises(ValueError, match="non-negative"):
        recommend_itemcf(teaching_similarity, np.zeros(3), np.zeros(3, dtype=bool), -1)


@pytest.mark.parametrize("observed", [np.array([[True, False, False]]), np.array([1, 0, 0])])
def test_observation_mask_requires_matching_boolean_vector(
    teaching_similarity: np.ndarray, observed: np.ndarray
) -> None:
    from memrecagent.itemcf import recommend_itemcf

    with pytest.raises(ValueError):
        recommend_itemcf(teaching_similarity, np.array([1, 0, 0]), observed, 2)


def test_positive_history_must_also_be_observed(teaching_similarity: np.ndarray) -> None:
    from memrecagent.itemcf import recommend_itemcf

    with pytest.raises(ValueError, match="observed"):
        recommend_itemcf(teaching_similarity, np.array([1, 0, 0]), np.zeros(3, dtype=bool), 2)
