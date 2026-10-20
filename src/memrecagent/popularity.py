"""Count positive training ratings and rank unseen candidates by popularity."""


def count_positive_ratings(
    train_records: list[tuple[str, str, float]], threshold: float = 4.0
) -> dict[str, int]:
    """Count positive users per item, retaining observed items with zero positives.

    Ratings are assumed to be valid 1–5 values, with a valid threshold supplied
    by the caller. Input must already be restricted to the training window.
    Duplicate user/item pairs are rejected: no timestamps are provided to
    resolve conflicting ratings. This function does not perform a time split.
    """
    popularity: dict[str, int] = {}
    seen_pairs: set[tuple[str, str]] = set()

    for user_id, item_id, rating in train_records:
        pair = (user_id, item_id)
        if pair in seen_pairs:
            raise ValueError("Duplicate user-item interaction")
        seen_pairs.add(pair)

        # 物品即使只有低评分，也保留在计数字典中，初始分数为 0。
        if item_id not in popularity:
            popularity[item_id] = 0
        if rating >= threshold:
            popularity[item_id] += 1

    return popularity


def recommend_popular(popularity: dict[str, int], seen_items: set[str], k: int) -> list[str]:
    """Filter seen items, rank by descending count then item ID, and take Top-K.

    The caller must provide non-negative counts computed from training data only.
    Dictionary keys define the available catalog; unseen catalog items absent
    from this dictionary are not added here. No user preference model is fitted.
    """
    if k < 0:
        raise ValueError("k must be non-negative")

    # 先过滤：已评分物品不能因为热门而重新进入候选。
    candidates = []
    for item in popularity:
        if item not in seen_items:
            candidates.append(item)

    # 再排序：分数降序；同分时按物品 ID 升序。
    ranked = sorted(candidates, key=lambda item: (-popularity[item], item))

    # 最后截取；候选不足 K 个时自然返回现有候选。
    return ranked[:k]
