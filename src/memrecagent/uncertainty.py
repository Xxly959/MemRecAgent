"""Paired user bootstrap for frozen recommendation results."""

from collections.abc import Mapping, Sequence, Set
from math import isfinite

import numpy as np

from memrecagent.metrics import evaluate_ranking, evaluate_users


def paired_recall_bootstrap(
    recommendations_a: Mapping[str, Sequence[str]],
    recommendations_b: Mapping[str, Sequence[str]],
    relevant: Mapping[str, Set[str]],
    k: int = 10,
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Percentile interval of mean per-user Recall A-B on fixed model outputs.

    Resampling paired differences is equivalent to giving both models the
    same user index draws. Users without positive targets are counted and
    excluded identically to evaluate_users; eligible zero-hit users stay.
    This does not resample interactions or retrain models. Approximate
    user-bootstrap inference assumes users are suitable exchangeable units.
    """
    if n_resamples <= 0 or not isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("Positive resample count and confidence in (0,1) required")
    first = evaluate_users(recommendations_a, relevant, k)
    second = evaluate_users(recommendations_b, relevant, k)
    if first.mean is None:
        raise ValueError("At least one user with positive relevant targets required")
    per_user = []
    for user in sorted(relevant):
        if not relevant[user]:
            continue
        a = evaluate_ranking(recommendations_a[user], relevant[user], k).recall
        b = evaluate_ranking(recommendations_b[user], relevant[user], k).recall
        per_user.append({"user_id": user, "recall_a": a, "recall_b": b, "difference": a - b})
    differences = np.array([row["difference"] for row in per_user], dtype=np.float64)
    rng = np.random.default_rng(seed)
    distribution = np.empty(n_resamples, dtype=np.float64)
    # Bound temporary memory while keeping one local continuing random stream.
    for start in range(0, n_resamples, 256):
        count = min(256, n_resamples - start)
        indices = rng.integers(0, len(differences), size=(count, len(differences)))
        distribution[start : start + count] = differences[indices].mean(axis=1)
    tail = (1 - confidence) / 2
    interval = np.quantile(distribution, [tail, 1 - tail], method="linear")
    return {
        "metric": "macro Recall@K",
        "k": k,
        "n_users": first.n_users,
        "n_evaluated": first.n_evaluated,
        "n_without_relevant": first.n_without_relevant,
        "mean_a": first.mean.recall,
        "mean_b": second.mean.recall,
        "delta_a_minus_b": float(differences.mean()),
        "interval": interval.tolist(),
        "confidence": confidence,
        "n_resamples": n_resamples,
        "seed": seed,
        "method": "paired user percentile bootstrap; numpy.quantile linear",
        "per_user": per_user,
        "bootstrap_deltas": distribution.tolist(),
    }
