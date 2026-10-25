"""固定完整候选重排，不允许截断TopK冒充权重搜索。"""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from memrecagent.ranking_sensitivity import evaluate_sensitivity, sensitivity_plan


def suite():
    frames = {}
    for norm in ("minmax", "rank"):
        frames[norm] = {}
        for group in ("G00", "G10", "G01", "G11"):
            frames[norm][group] = dict(
                status="ready",
                used_strategy="popularity",
                algorithm_effective=False,
                n_candidates=3,
                n_filtered=1,
                items=[
                    dict(
                        item_id="B",
                        base_score=80,
                        normalised_base_score=0.8 if norm == "minmax" else 0.5,
                        preference_signal=0,
                    ),
                    dict(
                        item_id="C",
                        base_score=0,
                        normalised_base_score=0,
                        preference_signal=int(group in {"G10", "G11"}),
                    ),
                ],
            )
    return dict(
        schema_version=1,
        evidence_kind="saved_response_replay",
        split="dev",
        cases=[
            dict(
                case_id="d:one",
                cluster_id="family-A",
                cohort="direct",
                reference_status="provisional",
                reference=[
                    dict(item_id="A", relevance=0, eligible=False),
                    dict(item_id="B", relevance=1, eligible=True),
                    dict(item_id="C", relevance=3, eligible=True),
                ],
                frames=frames,
            )
        ],
    )


def summary(report, norm="minmax", weight=0.75):
    return next(
        s
        for s in report["summaries"]
        if s["cohort"] == "direct"
        and s["reference_scope"] == "provisional"
        and s["normalisation"] == norm
        and s["weight"] == weight
    )


def test_full_candidate_scores_allow_reversal_and_rank_changes_breakpoint():
    s = suite()
    before = copy.deepcopy(s)
    report = evaluate_sensitivity(s, n_resamples=100, seed=42)
    rows = report["rows"]

    def ids(norm, weight):
        return next(
            r["groups"]["G10"]["item_ids"]
            for r in rows
            if r["normalisation"] == norm and r["weight"] == weight
        )

    assert ids("minmax", 0.75) == ["B", "C"]
    assert ids("minmax", 1) == ["C", "B"]
    assert ids("rank", 0.75) == ["C", "B"]
    assert summary(report)["groups"]["G10"]["ndcg"] == pytest.approx(0.7098097413968655)
    assert summary(report, "rank")["groups"]["G10"]["ndcg"] == 1
    assert summary(report, "rank")["paired_vs_default"]["G10"]["delta"] == pytest.approx(
        0.2901902586031345
    )
    assert report["api_calls"] == 0 and report["selected_configuration"] is None
    assert report["algorithm_effect"] is None and s == before


def test_failed_cases_stay_zero_and_reference_violations_not_mistaken_for_filter_success():
    s = suite()
    failed = copy.deepcopy(s["cases"][0])
    failed["case_id"] = "d:failed"
    for frame in failed["frames"].values():
        for arm in frame.values():
            arm.update(status="error", items=[], n_candidates=0, n_filtered=0, used_strategy=None)
    s["cases"].append(failed)
    s["cases"][0]["reference"][1]["eligible"] = False
    report = evaluate_sensitivity(s, n_resamples=20)
    group = summary(report)["groups"]["G10"]
    assert group["n_cases"] == 2 and group["n_errors"] == 1
    assert group["constraint_violations"] == 1
    assert group["ndcg"] == pytest.approx(0.31546487678572877)


def test_cluster_equal_weight_and_review_labels_cannot_choose_primary_winner():
    s = suite()
    original = s["cases"][0]
    # A族3例从BC变CB，B族1例从CB变BC；按族等权差0，按案例则错误为正。
    s["cases"] = [copy.deepcopy(original) for _ in range(4)]
    for i, c in enumerate(s["cases"]):
        c["case_id"] = f"d:{i}"
        c["cluster_id"] = "A" if i < 3 else "B"
    s["cases"][-1]["reference"][1]["relevance"] = 3
    s["cases"][-1]["reference"][2]["relevance"] = 1
    # B的default BC最佳，rank反转后变差；两族差应抵消。
    review = copy.deepcopy(original)
    review.update(case_id="d:review", reference_status="review_required", cluster_id="R")
    s["cases"].append(review)
    report = evaluate_sensitivity(s, n_resamples=20)
    g = summary(report)["groups"]["G10"]
    assert summary(report)["n_clusters"] == 2 and g["n_cases"] == 4
    assert g["ndcg"] == pytest.approx((0.7098097413968655 + 1) / 2)
    assert summary(report, "rank")["paired_vs_default"]["G10"]["delta"] == pytest.approx(0)
    assert len([r for r in report["rows"] if r["reference_status"] == "review_required"]) == 10


@pytest.mark.parametrize(
    "mutation",
    [
        "truncated",
        "duplicate",
        "changed_basis",
        "nonfinite",
        "missing_group",
        "test_split",
        "duplicate_case",
    ],
)
def test_untrusted_incomplete_or_mismatched_frames_rejected(mutation):
    s = suite()
    a = s["cases"][0]["frames"]["minmax"]["G10"]
    if mutation == "truncated":
        a["n_filtered"] = 0
    elif mutation == "duplicate":
        a["items"][1]["item_id"] = "B"
    elif mutation == "changed_basis":
        a["items"][0]["base_score"] = 81
    elif mutation == "nonfinite":
        a["items"][0]["normalised_base_score"] = float("nan")
    elif mutation == "missing_group":
        del s["cases"][0]["frames"]["rank"]["G01"]
    elif mutation == "test_split":
        s["split"] = "test"
    else:
        s["cases"].append(copy.deepcopy(s["cases"][0]))
    with pytest.raises(ValueError):
        sensitivity_plan(s)


def test_no_relevant_is_na_not_zero_and_unknown_algorithm_not_invented():
    s = suite()
    for r in s["cases"][0]["reference"]:
        r["relevance"] = 0
    r = evaluate_sensitivity(s, n_resamples=20)
    assert summary(r)["groups"]["G10"]["ndcg"] is None
    assert summary(r)["paired_vs_default"]["G10"] == dict(delta=None, interval=None, n_clusters=0)
    assert summary(r)["groups"]["G10"]["n_without_gain"] == 1


@pytest.mark.parametrize(
    "options", [{"n_resamples": True}, {"n_resamples": 10001}, {"seed": -1}, {"k": 0}]
)
def test_invalid_statistics_rejected_before_allocating_bootstrap(options):
    with pytest.raises(ValueError):
        evaluate_sensitivity(suite(), **options)


def test_cli_plan_no_artifact_then_run_no_overwrite_and_no_raw_error_echo(tmp_path):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(suite()))
    output = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "scripts/evaluate_ranking_sensitivity.py"
    args = [sys.executable, str(script), "--suite", str(path), "--output-dir", str(output)]
    plan = subprocess.run(args, capture_output=True, text=True)
    assert (
        plan.returncode == 0 and not output.exists() and json.loads(plan.stdout)["api_calls"] == 0
    )
    run = subprocess.run(args + ["--run"], capture_output=True, text=True)
    assert run.returncode == 0 and (output / "report.json").is_file()
    hashes = {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
    again = subprocess.run(args + ["--run"], capture_output=True, text=True)
    assert (
        again.returncode == 2 and again.stdout == "" and again.stderr == "sensitivity_unavailable\n"
    )
    assert hashes == {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
    path.write_text("private text must not echo")
    bad = subprocess.run(args, capture_output=True, text=True)
    assert bad.returncode == 2 and bad.stderr == "sensitivity_unavailable\n"
