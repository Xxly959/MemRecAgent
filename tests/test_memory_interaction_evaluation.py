"""连续场景评测的程序测试；合成客户端不代表模型理解能力。"""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_memory_interaction_extraction import envelope

from memrecagent.deepseek_client import ModelReply, ProviderError
from memrecagent.memory_interaction_evaluation import evaluate_interactions, load_interactions

FIXTURE = Path(__file__).parent / "fixtures/memory_interaction_cases_v1.json"


def model(raw):
    return ModelReply(
        json.dumps(raw, ensure_ascii=False), "fake", "synthetic", {"total_tokens": 10}, 0.1
    )


def scripted_client(cases, seen=None, change=None):
    # 仅测试客户端消费参考构造固定返回；产品运行器不能把参考送入complete。
    index = {}
    for episode in cases:
        for step in episode["steps"]:
            index[(episode["episode_id"], step["turn_id"])] = step["expected"]
            if step["reply"]:
                index[(episode["episode_id"], step["reply"]["turn_id"])] = step["reply"]["expected"]
    episode_index = -1

    def complete(messages):
        nonlocal episode_index
        payload = json.loads(messages[-1]["content"])
        if payload["latest_user_turn_id"] == "T1":
            episode_index += 1
        episode = cases[episode_index]["episode_id"]
        latest = payload["turns"][-1]
        expected = index[(episode, latest["turn_id"])]
        raw = envelope()
        raw["intents"] = expected["intents"]

        def quote(text=None, source=latest):
            return {"turn_id": source["turn_id"], "quote": text or source["content"]}

        def who(kind, person):
            refs = []
            if kind == "other":
                source = next(
                    t for t in payload["turns"] if t["role"] == "user" and person in t["content"]
                )
                refs = [quote(person, source)]
            return dict(kind=kind, references=refs)

        raw["topics"] = [quote(entity) for entity in expected["topics"]]
        for row in expected["assertions"]:
            raw["candidate"]["assertions"].append(
                dict(
                    subject=who(row["subject"], row["person"]),
                    category=row["category"],
                    entity=row["entity"],
                    relation=row["relation"],
                    scope=row["scope"],
                    evidence=[quote()],
                )
            )
        for row in expected["clues"]:
            raw["clues"].append(
                dict(
                    subject=who(row["subject"], row["person"]),
                    category=row["category"],
                    entity=row["entity"],
                    evidence=[quote()],
                )
            )
        request = expected["request"]
        if request["is_recommendation"]:
            raw["candidate"]["request"].update(
                is_recommendation=True,
                beneficiary=who(request["beneficiary"], request["person"]),
                category=request["category"],
                requested_entity=request["requested_entity"],
                change_scope=request["change_scope"],
                evidence=[quote()],
            )
        if expected["confirmation"] is not None:
            choice = expected["confirmation"]
            raw["confirmation"] = dict(
                question_turn_id=payload["pending_question"]["turn_id"],
                outcome=choice["outcome"],
                subject=choice["subject"],
                scope=choice["scope"],
                person_reference=None,
                evidence=[quote()],
            )
        if change:
            change(episode, latest, raw, payload)
        if seen is not None:
            seen.append(payload)
        return model(raw)

    return complete


def run(tmp_path, complete=None, cases=None):
    cases = cases or load_interactions(FIXTURE)
    return evaluate_interactions(
        cases, complete=complete or scripted_client(cases), output_dir=tmp_path / "run"
    )


def test_fixture_has_18_steps_two_conditional_replies_and_scripted_roles():
    cases = load_interactions(FIXTURE)
    assert len(cases) == 3 and sum(len(e["steps"]) for e in cases) == 18
    assert sum(s["reply"] is not None for e in cases for s in e["steps"]) == 2
    assert all(
        s["assistant"]["source"] == "scripted_assistant_context"
        for e in cases
        for s in e["steps"]
        if s["assistant"]
    )


def test_default_cli_no_key_no_directory_or_calls(tmp_path):
    env = os.environ.copy()
    env.pop("DEEPSEEK_API_KEY", None)
    target = tmp_path / "no-run"
    process = subprocess.run(
        [sys.executable, "scripts/evaluate_memory_interactions.py", "--output-dir", str(target)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    output = json.loads(process.stdout)
    assert output["mode"] == "plan" and output["api_calls"] == 0 and output["max_model_calls"] == 20
    assert not target.exists()


def test_complete_scripted_story_uses_actual_states_and_roles(tmp_path):
    cases = load_interactions(FIXTURE)
    seen = []
    report = run(tmp_path, scripted_client(cases, seen), cases)
    assert (
        report["model_calls"] == 20
        and report["strict_steps"] == 18
        and report["strict_episodes"] == 3
    )
    assert report["confirmation_correct"] == 2 and report["wrong_written_records"] == 0
    assert report["missing_updates"] == 0 and report["scripted_assistant_observed"] == 3
    assert report["heads_correct"] == 18 and report["activation_correct"] == 18
    text = json.dumps(seen, ensure_ascii=False)
    assert "allowed_writes" not in text and '"expected"' not in text and '"heads"' not in text
    assert "三星和苹果都可以进一步了解" in text
    events = [json.loads(line) for line in (tmp_path / "run/events.jsonl").read_text().splitlines()]
    assert sum(e["kind"] == "scripted_assistant_context" for e in events) == 3
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["max_calls"] == 20 and manifest["source_sha256"] and manifest["cases_sha256"]
    assert manifest["frozen_before_calls"] is True


def test_unasked_reply_and_wrong_action_script_never_appear(tmp_path):
    calls = []

    def complete(messages):
        calls.append(messages)
        raw = envelope()
        raw["intents"] = ["unrelated"]
        return model(raw)

    report = run(tmp_path, complete)
    text = json.dumps(calls, ensure_ascii=False)
    assert report["model_calls"] == 18 and report["confirmation_calls"] == 0
    assert report["confirmation_correct"] == 0 and report["conditional_not_seen"] == 2
    assert report["scripted_assistant_observed"] == 0
    assert "先只算这一次，不要长期保存" not in text
    assert "这是我自己的长期偏好" not in text
    assert "三星和苹果都可以进一步了解" not in text
    assert report["strict_steps"] < 18


def test_provider_error_stops_without_shrinking_denominator(tmp_path):
    def complete(messages):
        raise ProviderError("http", 401)

    report = run(tmp_path, complete)
    assert report["model_calls"] == 1 and report["not_run"] == 17
    assert report["base_steps"] == 18 and report["confirmation_correct"] == 0
    assert report["strict_episodes"] == 0 and report["provider_errors"] == 1


def test_budget_guard_is_before_sending(tmp_path, monkeypatch):
    import memrecagent.memory_interaction_evaluation as module

    monkeypatch.setattr(module, "MAX_CALLS", 1)
    calls = []

    def complete(messages):
        calls.append(messages)
        return model(envelope())

    report = run(tmp_path, complete)
    assert len(calls) == 1 and report["model_calls"] == 1 and report["budget_errors"] == 1


def test_existing_output_not_overwritten_before_any_call(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "marker").write_text("keep")
    calls = []
    with pytest.raises(FileExistsError):
        run(tmp_path, lambda m: calls.append(m))
    assert not calls and (output / "marker").read_text() == "keep"


def test_decode_error_is_counted_and_actual_prefix_kept(tmp_path):
    calls = []

    def complete(messages):
        calls.append(json.loads(messages[-1]["content"]))
        if len(calls) == 1:
            return ModelReply("{", "fake", None, None, 0.1)
        return model(envelope())

    report = run(tmp_path, complete)
    assert report["processing_errors"] >= 1 and report["model_calls"] == 18
    assert calls[1]["turns"][0]["content"] == "今天下雨了，我刚刚才回家。"
    assert report["strict_steps"] < 18


def test_wrong_other_person_and_wrong_writes_do_not_get_hidden_by_later_state(tmp_path):
    cases = load_interactions(FIXTURE)

    def change(episode, latest, raw, payload):
        if episode == "L_people_and_delete" and latest["turn_id"] == "T2":
            raw["candidate"]["assertions"][0]["subject"] = {"kind": "self", "references": []}

    report = run(tmp_path, scripted_client(cases, change=change), cases)
    rows = [json.loads(line) for line in (tmp_path / "run/steps.jsonl").read_text().splitlines()]
    wrong = next(
        r for r in rows if r["episode_id"] == "L_people_and_delete" and r["turn_id"] == "T2"
    )
    assert wrong["wrong_written_records"] == 1 and not wrong["checks"]["assertions"]
    assert report["wrong_written_records"] == 1 and report["strict_episodes"] < 3


@pytest.mark.parametrize("broken", ["episodes", "replies", "role", "reference"])
def test_invalid_frozen_inputs_rejected_before_calls_and_output(tmp_path, broken):
    cases = copy.deepcopy(load_interactions(FIXTURE))
    if broken == "episodes":
        cases = cases[:2]
    elif broken == "replies":
        cases[1]["steps"][2]["reply"] = None
    elif broken == "role":
        cases[0]["steps"][1]["assistant"]["source"] = "model_generated"
    else:
        cases[0]["steps"][0]["expected"]["assertions"] = [{"subject": "other", "person": None}]
    calls = []
    with pytest.raises(ValueError):
        run(tmp_path, lambda m: calls.append(m), cases)
    assert not calls and not (tmp_path / "run").exists()


def test_wrong_legal_person_quote_fails_person_and_activation_scores(tmp_path):
    cases = load_interactions(FIXTURE)

    def change(episode, latest, raw, payload):
        if episode == "L_people_and_delete" and latest["turn_id"] == "T4":
            raw["candidate"]["request"]["beneficiary"] = {
                "kind": "other",
                "references": [{"turn_id": latest["turn_id"], "quote": "给我"}],
            }

    report = run(tmp_path, scripted_client(cases, change=change), cases)
    rows = [json.loads(line) for line in (tmp_path / "run/steps.jsonl").read_text().splitlines()]
    row = next(r for r in rows if r["episode_id"] == "L_people_and_delete" and r["turn_id"] == "T4")
    assert not row["checks"]["subject"] and not row["checks"]["activation"]
    assert report["subject_correct"] < 18 and report["scope_correct"] == 18


def test_assistant_words_cannot_become_user_assertion_evidence(tmp_path):
    cases = load_interactions(FIXTURE)

    def change(episode, latest, raw, payload):
        if episode == "K_actual_questions" and latest["turn_id"] == "T2":
            assistant = next(t for t in payload["turns"] if t["role"] == "assistant")
            raw["intents"] = ["express_preference"]
            raw["candidate"]["assertions"] = [
                dict(
                    subject={"kind": "self", "references": []},
                    category="phone",
                    entity="三星",
                    relation="prefer",
                    scope="long_term",
                    evidence=[{"turn_id": assistant["turn_id"], "quote": assistant["content"]}],
                )
            ]

    report = run(tmp_path, scripted_client(cases, change=change), cases)
    rows = [json.loads(line) for line in (tmp_path / "run/steps.jsonl").read_text().splitlines()]
    row = next(r for r in rows if r["episode_id"] == "K_actual_questions" and r["turn_id"] == "T2")
    assert row["status"] == "processing_error" and not row["written"]
    assert report["processing_errors"] == 1


def test_callback_cannot_mutate_frozen_reference(tmp_path):
    cases = load_interactions(FIXTURE)
    correct = scripted_client(copy.deepcopy(cases))

    def complete(messages):
        response = correct(messages)
        cases[0]["steps"][0]["expected"]["heads"] = [["phone", "三星", "prefer"]]
        return response

    report = run(tmp_path, complete, cases)
    assert report["strict_steps"] == 18 and report["relation_correct"] == 18


def test_future_confirmation_reference_cannot_authorize_premature_long_term_write(tmp_path):
    cases = load_interactions(FIXTURE)

    def change(episode, latest, raw, payload):
        if episode == "K_actual_questions" and latest["turn_id"] == "T5":
            raw["candidate"]["assertions"][0]["scope"] = "long_term"

    report = run(tmp_path, scripted_client(cases, change=change), cases)
    rows = [json.loads(line) for line in (tmp_path / "run/steps.jsonl").read_text().splitlines()]
    row = next(r for r in rows if r["episode_id"] == "K_actual_questions" and r["turn_id"] == "T5")
    assert not row["reply_seen"] and row["written"] == [["phone", "苹果", "prefer"]]
    assert row["wrong_written_records"] == 1
    assert not row["checks"]["writes"] and not row["checks"]["heads"]
    assert report["model_calls"] == 19 and report["confirmation_reference_count"] == 2
    assert report["base_steps"] == 18 and report["confirmation_correct"] == 1
    assert report["wrong_written_records"] == 1
