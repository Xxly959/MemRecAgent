"""Aggregate fixed-protocol repeated training runs without selecting a seed."""

from math import isfinite
from statistics import mean, stdev


def summarize_runs(rows: list[dict]) -> dict:
    """Mean and sample standard deviation of all seeds, plus paired differences."""
    if len(rows) < 2 or len({row["seed"] for row in rows}) != len(rows):
        raise ValueError("At least two distinct seeds required; none may be discarded")
    result = {}
    for stage in ["validation", "test"]:
        cohort = None
        for row in rows:
            for model in ["mf", "bpr"]:
                metrics = row[stage][model]
                current = tuple(
                    metrics[key] for key in ["k", "n_users", "n_evaluated", "n_without_relevant"]
                )
                if cohort is not None and current != cohort:
                    raise ValueError("All seeds and models must use the same evaluation cohort")
                cohort = current
        result[stage] = {model: {} for model in ["mf", "bpr", "mf_minus_bpr"]}
        for metric in ["precision", "recall", "hit_rate", "ndcg"]:
            values = {}
            for model in ["mf", "bpr"]:
                try:
                    values[model] = [float(row[stage][model]["mean"][metric]) for row in rows]
                except (TypeError, KeyError) as exc:
                    raise ValueError("Complete numeric metrics required for every seed") from exc
                if not all(isfinite(value) for value in values[model]):
                    raise ValueError("Finite metrics required")
            values["mf_minus_bpr"] = [
                a - b for a, b in zip(values["mf"], values["bpr"], strict=True)
            ]
            for model, samples in values.items():
                result[stage][model][metric] = {
                    "n_seeds": len(samples),
                    "mean": mean(samples),
                    "std": stdev(samples),
                }
    return result
