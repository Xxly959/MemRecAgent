"""Small ItemCF building blocks introduced in the Phase 1 lessons."""

import numpy as np
from numpy.typing import NDArray


def item_cooccurrence(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Count shared positive users between item columns.

    Input is assumed to be a two-dimensional float32 binary user-item matrix.
    The diagonal counts each item's positive users; it is not normalized.
    The core expression below was supplied by the learner.
    """
    return x.T @ x


def cosine_item_similarity(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Fit dense item cosine similarities from a binary user-item matrix.

    Zero-norm items receive zero similarities, including on the diagonal.
    The caller is responsible for using only temporally valid training data.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not np.all((x == 0) | (x == 1)):
        raise ValueError("x must be a two-dimensional binary matrix")
    g = item_cooccurrence(x)
    norms = np.sqrt(np.diag(g))
    denominator = np.outer(norms, norms)
    # 只在分母大于 0 的位置相除，其他位置保留为 0，避免产生 NaN。
    return np.divide(g, denominator, out=np.zeros_like(g), where=denominator > 0)


def score_items(
    similarity: NDArray[np.float32], positive_history: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Sum item-similarity support from the user's binary positive history."""
    similarity = np.asarray(similarity, dtype=np.float32)
    positive_history = np.asarray(positive_history, dtype=np.float32)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be a square item-item matrix")
    if positive_history.ndim != 1 or positive_history.size != similarity.shape[0]:
        raise ValueError("positive_history must match the item dimension")
    if not np.all((positive_history == 0) | (positive_history == 1)):
        raise ValueError("positive_history must be binary")
    return positive_history @ similarity


def recommend_itemcf(
    similarity: NDArray[np.float32],
    positive_history: NDArray[np.float32],
    observed_history: NDArray[np.bool_],
    k: int,
) -> list[int]:
    """Return unseen item-column indices with strictly positive support.

    Filter every observed item, including low ratings. Break ties by column
    index; the caller maps indices back to item IDs. With insufficient support,
    return fewer than K items (or none), leaving any fallback to the caller.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    positive_history = np.asarray(positive_history, dtype=np.float32)
    observed_history = np.asarray(observed_history)
    if observed_history.ndim != 1 or observed_history.shape != positive_history.shape:
        raise ValueError("observed_history must match the history vector")
    if observed_history.dtype != np.bool_:
        raise ValueError("observed_history must have boolean dtype")
    if np.any((positive_history > 0) & ~observed_history):
        raise ValueError("Every positive history item must also be observed")
    scores = score_items(similarity, positive_history)
    candidates = np.flatnonzero(~observed_history & (scores > 0)).tolist()
    ranked = sorted(candidates, key=lambda column: (-float(scores[column]), column))
    return ranked[:k]
