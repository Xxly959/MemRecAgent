"""Independent numeric checks for seed aggregation, completeness and cohort consistency."""

import copy
import math

import pytest

from memrecagent.multiseed import summarize_runs


def seed_row(seed, mf, bpr):
    return {
        "seed": seed,
        **{
            stage: {
                model: {
                    "k": 10,
                    "n_users": 3,
                    "n_evaluated": 2,
                    "n_without_relevant": 1,
                    "mean": dict.fromkeys(["precision", "recall", "hit_rate", "ndcg"], value),
                }
                for model, value in [("mf", mf), ("bpr", bpr)]
            }
            for stage in ["validation", "test"]
        },
    }


def test_sample_standard_deviation_and_paired_seed_difference():
    rows = [seed_row(7, 1.0, 0.25), seed_row(8, 0.0, 0.25)]
    original = copy.deepcopy(rows)
    result = summarize_runs(rows)
    assert result["test"]["mf"]["recall"] == {
        "n_seeds": 2,
        "mean": 0.5,
        "std": pytest.approx(math.sqrt(0.5)),
    }
    assert result["test"]["bpr"]["recall"] == {"n_seeds": 2, "mean": 0.25, "std": 0.0}
    assert result["test"]["mf_minus_bpr"]["recall"]["mean"] == 0.25
    assert result["test"]["mf_minus_bpr"]["recall"]["std"] == pytest.approx(math.sqrt(0.5))
    assert rows == original


@pytest.mark.parametrize("seeds", [[], [7], [7, 7]])
def test_requires_two_distinct_seed_runs(seeds):
    with pytest.raises(ValueError):
        summarize_runs([seed_row(seed, 0.5, 0.25) for seed in seeds])


def test_different_user_cohorts_cannot_be_silently_averaged():
    rows = [seed_row(7, 0.5, 0.25), seed_row(8, 0.5, 0.25)]
    rows[1]["test"]["mf"]["n_evaluated"] = 1
    with pytest.raises(ValueError):
        summarize_runs(rows)


@pytest.mark.parametrize("bad", [None, float("nan")])
def test_missing_or_nonfinite_metrics_are_rejected(bad):
    rows = [seed_row(7, 0.5, 0.25), seed_row(8, 0.5, 0.25)]
    rows[1]["test"]["mf"]["mean"]["recall"] = bad
    with pytest.raises(ValueError):
        summarize_runs(rows)
