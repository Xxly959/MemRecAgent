"""派生政策与运行输入隔离，不能用修标签或位置捷径制造算法收益。"""

import json
from copy import deepcopy
from hashlib import sha256

import pytest
from test_external_evaluation import fixture

from memrecagent.external_derived import derive_external_suite


def digest(value):
    return sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def policy(**changes):
    return (
        dict(
            schema_version=1,
            policy_id="synthetic-policy-v1",
            seed=20261024,
            patches=[],
            turn_reviews=[],
        )
        | changes
    )


def derive():
    s, t = fixture()
    return derive_external_suite(s, t, policy())


def test_derivation_preserves_source_and_labels_do_not_enter_runtime():
    stories, tasks = fixture()
    before = deepcopy((stories, tasks))
    suite = derive_external_suite(stories, tasks, policy())
    assert (stories, tasks) == before
    task = suite["tasks"][0]
    assert set(task["runtime"]) == {"request_text", "items"}
    raw = json.dumps(task["runtime"])
    assert all(
        k not in raw for k in ("eligible", "relevance", "for_person", "hard_constraints", "reason")
    )
    assert task["reference"]["status"] == "provisional"
    assert suite["metadata"]["semantic_gold_independently_verified"] is False
    assert suite["metadata"]["algorithm_estimand_identified"] is False


def test_opaque_ids_are_stable_and_roundtrip_without_original_position_in_model_data():
    first, second = derive(), derive()
    assert first == second
    task = first["tasks"][0]
    ids = [x["item_id"] for x in task["runtime"]["items"]]
    assert len(ids) == len(set(ids)) == 2
    assert all(x.startswith("p_") and len(x) == 26 for x in ids)
    assert set(task["reference"]["original_ids"].values()) == {"A", "B"}
    assert {r["item_id"] for r in task["reference"]["item_scores"]} == set(ids)


def test_entire_scenario_family_stays_in_one_split_and_task_follows_story():
    s, t = fixture()
    all_stories, all_tasks = [], []
    for n in range(8):
        story, task = deepcopy(s[0]), deepcopy(t[0])
        story["story_id"] = task["story_id"] = f"S{n}"
        task["task_id"] = f"Q{n}"
        story["primary_scenario"] = f"family{n // 2}"
        for item, score in zip(task["items"], task["item_scores"], strict=True):
            item["item_id"] = score["item_id"] = f"old{n}-{item['item_id']}"
        all_stories.append(story)
        all_tasks.append(task)
    suite = derive_external_suite(all_stories, all_tasks, policy())
    by_family = {}
    for story in suite["episodes"]:
        by_family.setdefault(story["template_family"], set()).add(story["split"])
    assert all(len(v) == 1 for v in by_family.values())
    assert {e["split"] for e in suite["episodes"]} == {"dev", "test"}
    split = {e["episode_id"]: e["split"] for e in suite["episodes"]}
    assert all(task["split"] == split[task["episode_id"]] for task in suite["tasks"])


def test_current_budget_is_reference_override_not_runtime_configuration():
    stories, tasks = fixture()
    current = deepcopy(stories[0]["turns"][0]["gold"]["long_term_add"][0])
    current.update(value=5000, evidence_turn_ids=["T2"])
    stories[0]["turns"][1]["gold"]["request_only"] = [current]
    suite = derive_external_suite(stories, tasks, policy())
    ref = suite["tasks"][0]["reference"]
    assert ref["hard_constraints"][0]["value"] == 5000
    assert all(r["eligible"] for r in ref["item_scores"])
    assert (
        ref["status"] == "review_required" and "scoped_constraint_mismatch" in ref["review_codes"]
    )
    assert [r["relevance"] for r in ref["raw_item_scores"]] == [3, 0]
    assert "constraints" not in suite["tasks"][0]["runtime"]


def test_patch_requires_exact_old_value_and_logs_before_after_without_rewriting_text():
    stories, tasks = fixture()
    old = stories[0]["turns"][1]["gold"]["action"]
    patch = dict(
        story_id="X",
        turn_id="T2",
        field="action",
        before_sha256=digest(old),
        after="answer",
        reason="explicit policy example",
    )
    p = policy(patches=[patch])
    suite = derive_external_suite(stories, tasks, p)
    assert suite["episodes"][0]["steps"][1]["reference"]["action"] == "answer"
    assert suite["changes"][0]["before"] == old and suite["changes"][0]["after"] == "answer"
    assert stories[0]["turns"][1]["gold"]["action"] == old
    p["patches"][0]["before_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        derive_external_suite(stories, tasks, p)


def test_policy_cannot_patch_user_text_or_hide_unresolved_structural_error():
    stories, tasks = fixture()
    bad = dict(
        story_id="X",
        turn_id="T2",
        field="text",
        before_sha256=digest("推荐手机。"),
        after="new",
        reason="bad",
    )
    with pytest.raises(ValueError):
        derive_external_suite(stories, tasks, policy(patches=[bad]))
    stories[0]["turns"][0]["gold"]["long_term_add"][0]["evidence_turn_ids"] = ["future"]
    with pytest.raises(ValueError):
        derive_external_suite(stories, tasks, policy())


def test_only_immediate_same_session_reply_can_depend_on_clarification():
    stories, tasks = fixture()
    stories[0]["turns"][1]["gold"].update(needs_clarification=True, action="clarify")
    third = deepcopy(stories[0]["turns"][1])
    third.update(turn_id="T3", time="10:10", text="是我本人。")
    third["gold"].update(needs_clarification=False, action="answer")
    stories[0]["turns"].append(third)
    suite = derive_external_suite(stories, tasks, policy())
    assert suite["episodes"][0]["steps"][2]["reply_to"] == "T2"


def test_semantic_review_is_visible_and_never_becomes_verified_gold():
    p = policy(turn_reviews=[dict(story_id="X", turn_id="T1", reason="scope unclear")])
    stories, tasks = fixture()
    suite = derive_external_suite(stories, tasks, p)
    assert suite["episodes"][0]["steps"][0]["reference_reviews"] == ["scope unclear"]
    assert suite["tasks"][0]["reference"]["status"] == "review_required"
    assert suite["metadata"]["semantic_gold_independently_verified"] is False


def test_derivation_does_not_silently_repair_wrong_raw_eligibility():
    stories, tasks = fixture()
    tasks[0]["item_scores"][1].update(eligible=True, violated_constraints=[])
    with pytest.raises(ValueError):
        derive_external_suite(stories, tasks, policy())


def test_catalog_label_field_cannot_leak_through_runtime_items():
    stories, tasks = fixture()
    tasks[0]["items"][0]["relevance"] = 3
    with pytest.raises(ValueError):
        derive_external_suite(stories, tasks, policy())


@pytest.mark.parametrize("field,after", [("action", "invented"), ("needs_clarification", 1)])
def test_policy_cannot_create_invalid_reference_control_fields(field, after):
    stories, tasks = fixture()
    old = stories[0]["turns"][1]["gold"][field]
    patch = dict(
        story_id="X",
        turn_id="T2",
        field=field,
        before_sha256=digest(old),
        after=after,
        reason="invalid",
    )
    with pytest.raises(ValueError):
        derive_external_suite(stories, tasks, policy(patches=[patch]))


def test_same_preference_evidence_refresh_is_optional_not_required_new_fact():
    stories, tasks = fixture()
    repeated = deepcopy(stories[0]["turns"][0]["gold"]["long_term_add"][0])
    repeated["evidence_turn_ids"] = ["T1", "T2"]
    stories[0]["turns"][1]["gold"].update(
        long_term_add=[repeated], long_term_state_after=[repeated]
    )
    suite = derive_external_suite(stories, tasks, policy())
    transition = suite["episodes"][0]["steps"][1]["reference_transition"]
    assert transition["required_upserts"] == []
    assert transition["optional_refreshes"] == [repeated]


def test_source_story_identifiers_are_safe_for_later_per_user_store_paths():
    stories, tasks = fixture()
    stories[0]["story_id"] = tasks[0]["story_id"] = "../../escaped"
    with pytest.raises(ValueError):
        derive_external_suite(stories, tasks, policy())


def cli(tmp_path, *args):
    import subprocess
    import sys

    stories, tasks = fixture()
    for name, rows in (("stories", stories), ("tasks", tasks)):
        path = tmp_path / (name + ".jsonl")
        if not path.exists():
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    p = tmp_path / "policy.json"
    if not p.exists():
        p.write_text(json.dumps(policy()))
    return subprocess.run(
        [
            sys.executable,
            "scripts/derive_external_evaluation.py",
            "--stories",
            str(tmp_path / "stories.jsonl"),
            "--tasks",
            str(tmp_path / "tasks.jsonl"),
            "--policy",
            str(p),
            "--output-dir",
            str(tmp_path / "output"),
            *args,
        ],
        capture_output=True,
        timeout=15,
    )


def test_cli_plan_makes_no_files_then_explicit_derivation_freezes_raw_and_source(tmp_path):
    planned = cli(tmp_path)
    assert planned.returncode == 0, planned.stderr
    assert json.loads(planned.stdout)["api_calls"] == 0
    assert not (tmp_path / "output").exists()
    done = cli(tmp_path, "--derive")
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "output/raw/stories.jsonl").read_bytes() == (
        tmp_path / "stories.jsonl"
    ).read_bytes()
    manifest = json.loads((tmp_path / "output/manifest.json").read_text())
    assert manifest["api_calls"] == 0 and manifest["raw_inputs_unchanged"]
    assert any(k.endswith("external_derived.py") for k in manifest["source_sha256"])
    before = (tmp_path / "output/derived-suite.json").read_bytes()
    assert cli(tmp_path, "--derive").returncode == 2
    assert (tmp_path / "output/derived-suite.json").read_bytes() == before


def test_cli_invalid_policy_does_not_echo_source_or_leave_partial_dataset(tmp_path):
    (tmp_path / "policy.json").write_text(
        '{"private-phrase":"secret", "private-phrase":"duplicate"}'
    )
    result = cli(tmp_path, "--derive")
    assert result.returncode == 2
    assert json.loads(result.stderr) == {"error": "invalid_derivation_input"}
    assert b"private-phrase" not in result.stdout + result.stderr
    assert not (tmp_path / "output").exists()
