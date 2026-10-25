"""外部标注须独立核验，不能直接采信生成器的PASS报告。"""

import json
import os
import subprocess
import sys
from copy import deepcopy

from memrecagent.external_evaluation import audit_external_dataset


def fixture():
    fact = dict(
        subject="self",
        category="phone",
        target_type="attribute",
        attribute="price_cny",
        operator="<=",
        value=3000,
        unit="CNY",
        is_hard=True,
        evidence_turn_ids=["T1"],
    )
    gold = dict(
        long_term_add=[fact],
        long_term_retract=[],
        long_term_delete=[],
        long_term_state_after=[fact],
        session_only=[],
        request_only=[],
        action="answer",
        needs_clarification=False,
    )
    story = dict(
        story_id="X",
        primary_scenario="budget",
        people=[dict(person_id="self")],
        turns=[
            dict(
                turn_id="T1", session_id="S1", day=0, time="10:00", text="以后预算三千。", gold=gold
            ),
            dict(
                turn_id="T2",
                session_id="S2",
                day=1,
                time="10:00",
                text="推荐手机。",
                gold=deepcopy(gold) | dict(long_term_add=[], action="recommend"),
            ),
        ],
    )
    items = [
        dict(
            item_id="A",
            brand="甲",
            category="phone",
            price_cny=2000,
            weight_g=180,
            battery_life_hours=12,
            screen_inches=6,
        ),
        dict(
            item_id="B",
            brand="乙",
            category="phone",
            price_cny=4000,
            weight_g=180,
            battery_life_hours=12,
            screen_inches=6,
        ),
    ]
    task = dict(
        task_id="Q",
        story_id="X",
        turn_id="T2",
        request_text="推荐手机。",
        knowledge_cutoff_turn="T2",
        for_person="self",
        task_type="attribute",
        hard_constraints=[{k: fact[k] for k in ("attribute", "operator", "value", "unit")}],
        excluded_brands=[],
        items=items,
        item_scores=[
            dict(
                item_id="A", relevance=3, eligible=True, violated_constraints=[], reason="符合预算"
            ),
            dict(
                item_id="B",
                relevance=0,
                eligible=False,
                violated_constraints=["budget"],
                reason="超预算",
            ),
        ],
    )
    return [deepcopy(story)], [deepcopy(task)]


def codes(report):
    return {issue["code"] for issue in report["issues"]}


def test_audit_accepts_consistent_timeline_and_recomputes_eligibility():
    stories, tasks = fixture()
    result = audit_external_dataset(stories, tasks)
    assert result["structural_valid"]
    assert result["counts"] == dict(stories=1, turns=2, tasks=1, items=2, ratings=2)
    assert not result["issues"]


def test_audit_finds_future_evidence_even_when_reference_exists():
    stories, tasks = fixture()
    stories[0]["turns"][0]["gold"]["long_term_add"][0]["evidence_turn_ids"] = ["T2"]
    assert "future_or_missing_evidence" in codes(audit_external_dataset(stories, tasks))


def test_audit_recomputes_state_instead_of_trusting_state_after():
    stories, tasks = fixture()
    stories[0]["turns"][1]["gold"]["long_term_state_after"] = []
    assert "gold_state_mismatch" in codes(audit_external_dataset(stories, tasks))


def test_current_budget_overrides_long_term_and_must_not_be_intersected():
    stories, tasks = fixture()
    current = deepcopy(stories[0]["turns"][0]["gold"]["long_term_add"][0])
    current.update(value=5000, evidence_turn_ids=["T2"])
    stories[0]["turns"][1]["gold"]["request_only"] = [current]
    result = audit_external_dataset(stories, tasks)
    assert "scoped_constraint_mismatch" in codes(result)
    tasks[0]["hard_constraints"][0]["value"] = 5000
    tasks[0]["item_scores"][1].update(eligible=True, relevance=2, violated_constraints=[])
    assert not audit_external_dataset(stories, tasks)["issues"]


def test_wrong_eligibility_is_found_even_when_generator_claims_pass():
    stories, tasks = fixture()
    tasks[0]["item_scores"][1].update(eligible=True, violated_constraints=[])
    assert "eligibility_mismatch" in codes(audit_external_dataset(stories, tasks))


def test_unknown_attribute_and_boolean_numeric_value_fail_closed():
    for key, value in (("attribute", "mystery"), ("value", True)):
        stories, tasks = fixture()
        tasks[0]["hard_constraints"][0][key] = value
        result = audit_external_dataset(stories, tasks)
        assert not result["structural_valid"]
        assert "invalid_constraint" in codes(result)


def test_planned_denominator_survives_invalid_first_turn_schema():
    stories, tasks = fixture()
    del stories[0]["turns"][0]["gold"]["long_term_add"][0]["unit"]
    assert audit_external_dataset(stories, tasks)["counts"]["turns"] == 2


def test_ineligible_raw_relevance_can_be_positive_but_requires_masked_utility():
    stories, tasks = fixture()
    tasks[0]["item_scores"][1]["relevance"] = 2
    report = audit_external_dataset(stories, tasks)
    assert not report["issues"]
    assert report["ineligible_positive_raw_ratings"] == 1
    assert report["ranking_gain_policy"] == "eligible_only"


def test_storage_numeric_and_screen_enum_are_valid_typed_attributes():
    stories, tasks = fixture()
    for attribute, op, value, unit in (
        ("storage_gb", ">=", 512, "GB"),
        ("screen_type", "=", "OLED", "type"),
    ):
        fact = dict(
            subject="self",
            category="phone",
            target_type="attribute",
            attribute=attribute,
            operator=op,
            value=value,
            unit=unit,
            is_hard=False,
            evidence_turn_ids=["T1"],
        )
        stories[0]["turns"][0]["gold"]["long_term_add"].append(fact)
        for turn in stories[0]["turns"]:
            turn["gold"]["long_term_state_after"].append(fact)
    assert audit_external_dataset(stories, tasks)["structural_valid"]


def test_enum_hard_constraint_uses_equality_not_numeric_comparison():
    stories, tasks = fixture()
    for turn in stories[0]["turns"]:
        for field in ("long_term_add", "long_term_state_after"):
            for fact in turn["gold"][field]:
                fact.update(attribute="screen_type", operator="=", value="OLED", unit="type")
    tasks[0]["hard_constraints"] = [
        dict(attribute="screen_type", operator="=", value="OLED", unit="type")
    ]
    for item, value in zip(tasks[0]["items"], ("OLED", "LCD"), strict=True):
        item["screen_type"] = value
    assert not audit_external_dataset(stories, tasks)["issues"]


def test_unconstrained_item_numeric_metadata_still_rejects_nonfinite_values():
    stories, tasks = fixture()
    tasks[0]["items"][0]["weight_g"] = float("inf")
    assert not audit_external_dataset(stories, tasks)["structural_valid"]


def test_duplicate_score_and_wrong_owner_long_term_are_not_silently_collapsed():
    stories, tasks = fixture()
    tasks[0]["item_scores"].append(deepcopy(tasks[0]["item_scores"][0]))
    stories[0]["turns"][0]["gold"]["long_term_add"][0]["subject"] = "mother"
    result = audit_external_dataset(stories, tasks)
    assert {"duplicate_rating_id", "invalid_long_term_owner"} <= codes(result)


def test_different_grade_without_expressed_attribute_requires_review_not_auto_repair():
    stories, tasks = fixture()
    for turn in stories[0]["turns"]:
        turn["gold"].update(long_term_add=[], long_term_state_after=[])
    tasks[0]["hard_constraints"] = []
    tasks[0]["items"][1]["brand"] = "甲"
    tasks[0]["item_scores"][1].update(eligible=True, relevance=1, violated_constraints=[])
    original = deepcopy(tasks)
    result = audit_external_dataset(stories, tasks)
    assert "within_brand_grade_needs_review" in codes(result)
    assert result["structural_valid"] and result["human_review_required"]
    assert tasks == original


def run_cli(tmp_path, stories=None, tasks=None):
    s, t = fixture()
    for name, rows in (
        ("stories", s if stories is None else stories),
        ("tasks", t if tasks is None else tasks),
    ):
        path = tmp_path / (name + ".jsonl")
        if not path.exists():
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    env = {k: v for k, v in os.environ.items() if k != "DEEPSEEK_API_KEY"}
    return subprocess.run(
        [
            sys.executable,
            "scripts/audit_external_evaluation.py",
            "--stories",
            str(tmp_path / "stories.jsonl"),
            "--tasks",
            str(tmp_path / "tasks.jsonl"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
        capture_output=True,
        env=env,
    )


def test_cli_preserves_exact_raw_bytes_and_refuses_overwrite(tmp_path):
    result = run_cli(tmp_path)
    assert result.returncode == 0, result.stderr
    output = tmp_path / "output"
    assert (output / "stories.jsonl").read_bytes() == (tmp_path / "stories.jsonl").read_bytes()
    assert json.loads((output / "manifest.json").read_text())["api_calls"] == 0
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    assert run_cli(tmp_path).returncode == 2
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


def test_cli_rejects_duplicate_json_keys_without_echoing_source(tmp_path):
    (tmp_path / "stories.jsonl").write_text('{"secret":"private-content","secret":"x"}\n')
    result = run_cli(tmp_path)
    assert result.returncode == 2
    assert b"private-content" not in result.stdout + result.stderr
    assert json.loads(result.stderr)["error"] == "invalid_external_input"
    assert not (tmp_path / "output").exists()
