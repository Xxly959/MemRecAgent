"""Cold-user routing between existing, filtered recommendation lists."""

from collections.abc import Sequence, Set


def recommend_with_fallback(
    user_id: str,
    history_users: Set[str],
    bpr_items: Sequence[str],
    popular_items: Sequence[str],
    k: int,
) -> list[str]:
    """Use popularity only for absent history users; scores/targets are not inputs.

    Caller supplies sorted lists from the same legal catalog, already filtered
    for this user. Presence in history, not positive-history count or a score
    threshold, defines this deliberately narrow cold-start policy.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    selected = popular_items if user_id not in history_users else bpr_items
    if len(set(selected)) != len(selected):
        raise ValueError("Selected ranking must not contain duplicate items")
    return list(selected[:k])
