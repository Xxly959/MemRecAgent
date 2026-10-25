"""独立手算期望；防止漏失败、删候选外答案、按对话轮伪造样本量。"""

from copy import deepcopy

import pytest


def observation(case="c1", cluster="u1", *, ranks=None, relevant=None):
    ranks = ranks or {"G00": ["A"], "G10": ["B"], "G01": ["B"], "G11": ["B"]}
    return {
        "case_id": case,
        "cluster_id": cluster,
        "stratum": "warm",
        "relevant": ["B"] if relevant is None else relevant,
        "excluded": ["A"],
        "groups": {
            name: {
                "item_ids": ids,
                "status": "ready",
                "algorithm_effective": name in {"G01", "G11"},
            }
            for name, ids in ranks.items()
        },
    }


def test_four_way_interaction_is_not_confused_with_beating_two_baselines():
    from memrecagent.memory_ablation_metrics import summarize_ablation

    result = summarize_ablation([observation()], k=1, n_resamples=100, seed=42)
    assert result["groups"]["G00"]["ndcg"] == 0
    assert result["groups"]["G11"]["ndcg"] == 1
    assert result["contrasts"]["interaction"]["delta"] == -1
    assert result["contrasts"]["interaction"]["interval"] == [-1, -1]
    assert result["groups"]["G00"]["n_constraint_violations"] == 1
    assert result["groups"]["G11"]["constraint_violation_rate"] == 0
    assert result["n_clusters"] == 1


def test_users_receive_equal_weight_despite_different_number_of_turns():
    from memrecagent.memory_ablation_metrics import summarize_ablation

    bad = {g: ["A"] for g in ("G00", "G10", "G01", "G11")}
    good = {g: ["B"] for g in bad}
    rows = [observation(f"c{i}", "u1", ranks=bad) for i in range(3)]
    rows.append(observation("c4", "u2", ranks=good))
    result = summarize_ablation(rows, k=1, n_resamples=100, seed=7)
    assert result["n_cases"] == 4 and result["n_clusters"] == 2
    assert result["groups"]["G11"]["ndcg"] == 0.5
    assert result["groups"]["G11"]["case_mean_ndcg"] == 0.25
    assert result["contrasts"]["memory_given_algorithm"]["interval"] == [0, 0]


def test_outside_candidate_target_remains_in_recall_denominator_and_errors_count():
    from memrecagent.memory_ablation_metrics import summarize_ablation

    row = observation(relevant=["B", "OUTSIDE"])
    row["groups"]["G11"] = {"item_ids": [], "status": "error", "algorithm_effective": False}
    result = summarize_ablation([row], k=1, n_resamples=100, seed=1)
    assert result["groups"]["G10"]["recall"] == 0.5
    assert result["groups"]["G11"]["recall"] == 0
    assert result["groups"]["G11"]["n_errors"] == 1
    assert result["groups"]["G11"]["empty_rate"] == 1
    assert result["n_effective_algorithm_cases"] == 0


def test_no_relevant_is_null_not_a_fake_zero_and_constraints_still_evaluated():
    from memrecagent.memory_ablation_metrics import summarize_ablation

    result = summarize_ablation([observation(relevant=[])], k=1, n_resamples=100, seed=42)
    assert result["n_without_relevant"] == 1
    assert result["groups"]["G11"]["ndcg"] is None
    assert result["contrasts"]["interaction"]["interval"] is None
    assert result["groups"]["G00"]["constraint_violation_rate"] == 1


def test_seed_repeatability_and_input_not_mutated():
    from memrecagent.memory_ablation_metrics import summarize_ablation

    rows = [observation(), observation("c2", "u2")]
    before = deepcopy(rows)
    assert summarize_ablation(rows, k=1, n_resamples=100, seed=9) == summarize_ablation(
        rows, k=1, n_resamples=100, seed=9
    )
    assert rows == before


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_case",
        "missing_group",
        "duplicate_item",
        "error_items",
        "unknown_group",
        "no_cluster",
        "invalid_status",
        "bad_flag",
    ],
)
def test_invalid_observations_fail_instead_of_silently_changing_denominators(mutation):
    from memrecagent.memory_ablation_metrics import summarize_ablation

    rows = [observation()]
    if mutation == "duplicate_case":
        rows.append(observation())
    elif mutation == "missing_group":
        del rows[0]["groups"]["G00"]
    elif mutation == "duplicate_item":
        rows[0]["groups"]["G00"]["item_ids"] = ["A", "A"]
    elif mutation == "error_items":
        rows[0]["groups"]["G00"]["status"] = "error"
    elif mutation == "unknown_group":
        rows[0]["groups"]["other"] = rows[0]["groups"]["G00"]
    elif mutation == "no_cluster":
        rows[0]["cluster_id"] = ""
    elif mutation == "invalid_status":
        rows[0]["groups"]["G00"]["status"] = "ok"
    elif mutation == "bad_flag":
        rows[0]["groups"]["G00"]["algorithm_effective"] = "false"
    with pytest.raises(ValueError):
        summarize_ablation(rows, k=1, n_resamples=100, seed=42)


@pytest.mark.parametrize(
    "options", [{"k": True}, {"k": 0}, {"n_resamples": 0}, {"n_resamples": True}, {"seed": True}]
)
def test_invalid_statistical_options_fail(options):
    from memrecagent.memory_ablation_metrics import summarize_ablation

    with pytest.raises(ValueError):
        summarize_ablation([observation()], **({"k": 1, "n_resamples": 100, "seed": 42} | options))
