"""Adapt existing baselines to one fixed candidate and relevance protocol."""

from dataclasses import asdict

import numpy as np

from memrecagent.interactions import build_user_item_matrix
from memrecagent.itemcf import cosine_item_similarity, score_items
from memrecagent.metrics import evaluate_users
from memrecagent.movielens import Rating
from memrecagent.popularity import count_positive_ratings, recommend_popular


def evaluate_holdout(
    history: list[Rating], holdout: list[Rating], threshold: float = 4.0, k: int = 10
) -> dict:
    """Fit on history, freeze scores, then evaluate the later holdout window.

    Catalog comes solely from history. Both baselines rank all unseen catalog
    items, including zero scores, with string-ID tie breaks. Unlike the toy
    recommend_itemcf helper, this adapter does not restrict to positive scores.
    All held-out positives remain in the denominator, even outside catalog.
    Cold users receive zero ItemCF scores; there is no popularity fallback.
    """
    if not history or not 1 <= threshold <= 5 or k <= 0:
        raise ValueError("Nonempty history, threshold in [1,5], and positive k required")
    if holdout and max(r.timestamp for r in history) >= min(r.timestamp for r in holdout):
        raise ValueError("All history must strictly precede the holdout window")
    pairs = set()
    for record in [*history, *holdout]:
        pair = (record.user_id, record.item_id)
        if pair in pairs:
            raise ValueError("Duplicate user-item interaction")
        pairs.add(pair)

    train_records = sorted((r.user_id, r.item_id, r.rating) for r in history)
    ratings, observed, users, items = build_user_item_matrix(train_records)
    positive = (observed & (ratings >= threshold)).astype(np.float32)
    similarity = cosine_item_similarity(positive)
    popularity = count_positive_ratings(train_records, threshold)
    catalog = sorted(items)
    cohort = sorted({r.user_id for r in holdout})
    relevant: dict[str, set[str]] = {user: set() for user in cohort}
    for record in holdout:
        if record.rating >= threshold:
            relevant[record.user_id].add(record.item_id)

    predictions: dict[str, dict[str, list[str]]] = {"popularity": {}, "itemcf": {}}
    user_stats = {}
    for user in cohort:
        row = users.get(user)
        if row is None:
            seen = set()
            h = np.zeros(len(items), dtype=np.float32)
        else:
            seen = {item for item in catalog if observed[row, items[item]]}
            h = positive[row]
        scores = score_items(similarity, h)
        candidates = [item for item in catalog if item not in seen]
        predictions["popularity"][user] = recommend_popular(popularity, seen, k)
        predictions["itemcf"][user] = sorted(
            candidates, key=lambda item: (-float(scores[items[item]]), item)
        )[:k]
        user_stats[user] = {
            "cold_user": row is None,
            "n_seen": len(seen),
            "n_positive_history": int(h.sum()),
            "n_candidates": len(candidates),
            "n_relevant": len(relevant[user]),
            "n_relevant_outside_catalog": len(relevant[user] - items.keys()),
        }

    return {
        "stats": {
            "n_history_ratings": len(history),
            "n_holdout_ratings": len(holdout),
            "n_history_users": len(users),
            "n_catalog_items": len(items),
            "n_holdout_users": len(cohort),
            "n_cold_users": sum(s["cold_user"] for s in user_stats.values()),
            "n_cold_users_with_relevant": sum(
                s["cold_user"] and s["n_relevant"] > 0 for s in user_stats.values()
            ),
            "n_users_without_positive_history": sum(
                s["n_positive_history"] == 0 for s in user_stats.values()
            ),
            "n_positive_targets": sum(len(answers) for answers in relevant.values()),
            "n_positive_targets_outside_catalog": sum(
                s["n_relevant_outside_catalog"] for s in user_stats.values()
            ),
            "n_holdout_items_outside_catalog": len({r.item_id for r in holdout} - items.keys()),
        },
        "metrics": {
            model: asdict(evaluate_users(ranked, relevant, k))
            for model, ranked in predictions.items()
        },
        "predictions": predictions,
        "relevant": {user: sorted(answers) for user, answers in relevant.items()},
        "users": user_stats,
    }
