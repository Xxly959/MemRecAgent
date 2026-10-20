"""Convert explicit rating records into small matrices for Phase 1 lessons."""

import numpy as np
from numpy.typing import NDArray


def build_user_item_matrix(
    interactions: list[tuple[str, str, float]],
) -> tuple[NDArray[np.float32], NDArray[np.bool_], dict[str, int], dict[str, int]]:
    """Build ratings, observation mask, and ID mappings in first-seen order.

    Ratings are assumed to be valid 1–5 values for this teaching exercise.
    Missing cells store zero but are distinguished by the boolean mask.
    Duplicate user/item pairs raise ValueError because records have no timestamps.
    """
    user_to_index: dict[str, int] = {}
    item_to_index: dict[str, int] = {}

    # 第一步：把用户和物品名称对应到连续的行号、列号。
    for user_id, item_id, _rating in interactions:
        if user_id not in user_to_index:
            user_to_index[user_id] = len(user_to_index)
        if item_id not in item_to_index:
            item_to_index[item_id] = len(item_to_index)

    # 第二步：知道行列数量后，创建空的评分矩阵和观测掩码。
    shape = (len(user_to_index), len(item_to_index))
    ratings = np.zeros(shape, dtype=np.float32)
    observed = np.zeros(shape, dtype=bool)

    # 第三步：查出每条记录的行列位置，填评分并标记“有观测”。
    for user_id, item_id, rating in interactions:
        row = user_to_index[user_id]
        column = item_to_index[item_id]
        if observed[row, column]:
            raise ValueError("Duplicate user-item interaction")
        ratings[row, column] = rating
        observed[row, column] = True

    return ratings, observed, user_to_index, item_to_index
