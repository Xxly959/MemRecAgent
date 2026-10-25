"""验证模型输入轮次边界与固定套件选择，不用提示文字匹配代替模型实验。"""

import json
import os
import runpy
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest
from test_memory_confirmation import answer, data
from test_memory_extraction import NOW, output_for, turns_for

from memrecagent.memory_confirmation import decode_confirmation
from memrecagent.memory_extraction import ExtractionError, Turn, decode_candidate
from memrecagent.memory_extraction_evaluation import build_messages


@pytest.mark.parametrize("error", ["attitude", "request_target", "correct"])
def test_strict_score_rejects_wrong_attitude_or_request_target(tmp_path, error):
    from test_memory_dialogue import empty, model

    from memrecagent.memory_dialogue import evaluate_dialogues, load_dialogues

    cases = load_dialogues(Path("tests/fixtures/memory_dialogue_cases.json"))
    cases = json.loads(json.dumps(cases))
    first = cases[0]["steps"][0]
    first["text"] = "我妹妹长期喜欢苹果，这次给我看看小米手机。"
    first["expected"] = {
        "actions": ["current"],
        "heads": [],
        "active": [],
        "excluded": [],
        "beneficiary": "self",
        "allowed_writes": [],
        "person": None,
        "semantic": {
            "request": {
                "is_recommendation": True,
                "beneficiary": "self",
                "person": None,
                "category": "phone",
                "requested_entity": "小米",
                "change_scope": "request",
            },
            "assertions": [
                {
                    "subject": "other",
                    "person": "我妹妹",
                    "category": "phone",
                    "entity": "苹果",
                    "relation": "prefer",
                    "scope": "long_term",
                }
            ],
        },
    }

    def complete(messages):
        payload = json.loads(messages[1]["content"])
        turn = payload["turns"][-1]
        value = empty()
        if turn["content"] == first["text"]:
            evidence = [{"turn_id": turn["turn_id"], "start": 0, "end": len(turn["content"])}]
            value["request"].update(
                is_recommendation=True,
                beneficiary={"kind": "self", "references": []},
                category="phone",
                requested_entity="手机" if error == "request_target" else "小米",
                evidence=evidence,
            )
            value["assertions"] = [
                {
                    "subject": {
                        "kind": "other",
                        "references": [{"turn_id": turn["turn_id"], "start": 0, "end": 3}],
                    },
                    "category": "phone",
                    "entity": "苹果",
                    "relation": "dislike" if error == "attitude" else "prefer",
                    "scope": "long_term",
                    "evidence": evidence,
                }
            ]
        return model(value)

    report = evaluate_dialogues(tuple(cases), complete=complete, output_dir=tmp_path / "run")
    first_result = json.loads((tmp_path / "run/steps.jsonl").read_text().splitlines()[0])
    assert first_result["checks"]["writes"] is True
    assert first_result["strict"] is (error == "correct")
    assert report["semantic_scored_steps"] == 1
    assert report["semantic_correct"] == (1 if error == "correct" else 0)


def test_live_cli_actually_loads_new_suite_under_offline_failure(tmp_path, monkeypatch, capsys):
    from memrecagent.deepseek_client import DeepSeekClient, ProviderError

    def fail_offline(self, messages):
        raise ProviderError("response_protocol")

    monkeypatch.setattr(DeepSeekClient, "complete", fail_offline)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "synthetic-not-a-credential")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_memory_dialogues.py",
            "--suite",
            "acceptance-v2",
            "--live",
            "--output-dir",
            str(tmp_path / "run"),
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path("scripts/evaluate_memory_dialogues.py", run_name="__main__")
    assert exit_info.value.code == 1
    inputs = json.loads((tmp_path / "run/inputs.json").read_text())
    assert [e["episode_id"] for e in inputs] == [
        "D_current_history_v2",
        "E_subject_scope_v2",
        "F_age_delete_v2",
    ]
    assert json.loads(capsys.readouterr().out)["report"]["not_run"] == 17


def test_input_explicitly_identifies_latest_user_turn_after_history():
    older = Turn("old", "user", "我姐喜欢一加手机。", NOW, NOW)
    assistant = Turn("question", "assistant", "这次为谁选？", NOW, NOW)
    current = Turn(
        "new", "user", "给我看荣耀手机。", NOW + timedelta(seconds=1), NOW + timedelta(seconds=1)
    )
    messages = build_messages((older, assistant, current), ())
    payload = json.loads(messages[1]["content"])
    assert payload["latest_user_turn_id"] == "new"
    assert [t["turn_id"] for t in payload["turns"]] == ["old", "question", "new"]
    assert payload["own_memories"] == []


def test_new_suite_plan_selects_fixed_synthetic_cases_without_key_or_output(tmp_path):
    env = os.environ.copy()
    env.pop("DEEPSEEK_API_KEY", None)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_memory_dialogues.py",
            "--suite",
            "acceptance-v2",
            "--output-dir",
            str(tmp_path / "run"),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "mode": "plan",
        "suite": "acceptance-v2",
        "episodes": 3,
        "base_steps": 18,
        "max_model_calls": 20,
        "api_calls": 0,
    }
    assert not (tmp_path / "run").exists()


def test_new_suite_live_without_key_stops_before_experiment_creation(tmp_path):
    env = os.environ.copy()
    env.pop("DEEPSEEK_API_KEY", None)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_memory_dialogues.py",
            "--suite",
            "acceptance-v2",
            "--live",
            "--output-dir",
            str(tmp_path / "run"),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 2
    assert "DEEPSEEK_API_KEY" in result.stderr
    assert not (tmp_path / "run").exists()


def test_request_current_alias_stays_invalid_after_prompt_fix():
    turns = turns_for("这次给我看小米手机。")
    value = output_for(turns[0])
    value["request"]["change_scope"] = "current"
    with pytest.raises(ExtractionError, match="enum"):
        decode_candidate(json.dumps(value), turns=turns)


def test_confirmation_person_string_is_never_silently_converted_to_span():
    reply = answer("是我哥，这次给我哥挑。")
    value = data(reply, subject="other", person_reference="我哥")
    with pytest.raises(ExtractionError, match="confirmation_evidence"):
        decode_confirmation(json.dumps(value), reply=reply)
