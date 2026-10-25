"""双阶段新验收的离线运行器测试，不代表模型语义通过。"""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from test_memory_interaction_evaluation import scripted_client

from memrecagent.deepseek_client import ModelReply, ProviderError
from memrecagent.memory_interaction_evaluation import (
    _hash,
    evaluate_interactions,
    load_interactions,
)

FIXTURE = Path(__file__).parent / "fixtures/memory_analyst_cases_v1.json"


def paired_client(cases, seen=None):
    delegate = scripted_client(cases)
    previous = None
    references = {step["text"]: step["expected"] for ep in cases for step in ep["steps"]}

    def complete(messages):
        nonlocal previous
        payload = json.loads(messages[-1]["content"])
        if seen is not None:
            seen.append(payload)
        if "memory_view" not in payload:
            previous = delegate(messages)
            # 旧测试客户端未处理显式unresolved；仅在本固定客户端补齐真实完整响应。
            latest = payload["turns"][-1]
            reference = references.get(latest["content"])
            if reference is not None:
                data = json.loads(previous.content)
                data["candidate"]["unresolved"] = [
                    {
                        "field": name,
                        "evidence": [{"turn_id": latest["turn_id"], "quote": latest["content"]}],
                    }
                    for name in reference["unresolved"]
                ]
                previous = replace(previous, content=json.dumps(data, ensure_ascii=False))
        return previous

    return complete


def run(tmp_path, mode="analyst", complete=None):
    cases = load_interactions(FIXTURE)
    return evaluate_interactions(
        cases,
        complete=complete or paired_client(cases),
        output_dir=tmp_path / "run",
        analysis_mode=mode,
    )


@pytest.mark.parametrize("mode,count,confirm_calls", [("direct", 20, 2), ("analyst", 40, 4)])
def test_complete_new_story_counts_physical_calls_and_logical_confirmations(
    tmp_path, mode, count, confirm_calls
):
    cases = load_interactions(FIXTURE)
    seen = []
    result = run(tmp_path, mode, paired_client(cases, seen))
    assert result["strict_steps"] == 18 and result["strict_episodes"] == 3
    assert result["model_calls"] == count
    assert result["confirmation_calls"] == confirm_calls
    assert result["confirmation_turns"] == 2 and result["confirmation_correct"] == 2
    assert result["wrong_written_records"] == 0 and result["missing_updates"] == 0
    assert result["scripted_assistant_observed"] == 3
    assert result["stage_calls"] == (
        {"direct": 20, "draft": 0, "analyst": 0}
        if mode == "direct"
        else {"direct": 0, "draft": 20, "analyst": 20}
    )
    assert '"expected"' not in json.dumps(seen) and '"writes"' not in json.dumps(seen)
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["analysis_mode"] == mode and manifest["max_calls"] == count
    assert manifest["analyst_prompt_sha256"] and manifest["dependency_sha256"]
    assert manifest["frozen_before_calls"] is True


@pytest.mark.parametrize("failing_stage", [1, 2])
def test_provider_failure_stops_all_following_steps_without_losing_denominator(
    tmp_path, failing_stage
):
    cases = load_interactions(FIXTURE)
    fixed = paired_client(cases)
    calls = []

    def complete(messages):
        calls.append(messages)
        if len(calls) == failing_stage:
            raise ProviderError("http", 429)
        return fixed(messages)

    report = run(tmp_path, complete=complete)
    assert len(calls) == failing_stage and report["model_calls"] == failing_stage
    assert report["provider_errors"] == 1 and report["not_run"] == 17
    assert report["base_steps"] == 18 and report["confirmation_reference_count"] == 2
    assert report["strict_episodes"] == 0 and report["confirmation_correct"] == 0


def test_two_call_budget_is_checked_before_starting_an_incomplete_pair(tmp_path, monkeypatch):
    import memrecagent.memory_interaction_evaluation as module

    monkeypatch.setattr(module, "MAX_CALLS", 1)
    seen = []
    report = run(tmp_path, complete=paired_client(load_interactions(FIXTURE), seen))
    assert report["model_calls"] == 2 and len(seen) == 2
    assert report["budget_errors"] == 1 and report["not_run"] == 16
    assert report["stage_calls"] == {"direct": 0, "draft": 1, "analyst": 1}


def test_bad_draft_runs_independent_final_and_keeps_rejected_attempt(tmp_path):
    cases = load_interactions(FIXTURE)
    fixed = paired_client(cases)
    seen = []

    def complete(messages):
        seen.append(json.loads(messages[-1]["content"]))
        response = fixed(messages)
        return ModelReply("{", "fixed", None, None, 0) if len(seen) == 1 else response

    report = run(tmp_path, complete=complete)
    assert report["processing_errors"] == 0 and report["model_calls"] == 40
    assert seen[1]["turns"][0]["turn_id"] == "T1"
    assert seen[1]["latest_user_turn_id"] == "T1" and "memory_view" in seen[1]
    assert seen[1]["untrusted_draft"]["raw"] == "{"
    assert report["draft_processing_errors"] == 1
    assert report["strict_steps"] == 18


@pytest.mark.parametrize("mode", ["bad", None, True])
def test_invalid_mode_rejected_before_calls_or_output(tmp_path, mode):
    with pytest.raises(ValueError):
        run(tmp_path, mode, lambda _: pytest.fail("不得调用"))
    assert not (tmp_path / "run").exists()


def test_saved_request_hash_and_messages_cannot_be_mutated_by_callback(tmp_path):
    fixed = paired_client(load_interactions(FIXTURE))

    def complete(messages):
        response = fixed(messages)
        messages.clear()
        return response

    report = run(tmp_path, complete=complete)
    calls = [json.loads(s) for s in (tmp_path / "run/calls.jsonl").read_text().splitlines()]
    assert report["strict_steps"] == 18
    assert len(calls) == 40
    assert all(
        len(c["messages"]) == 2 and _hash(c["messages"]) == c["request_sha256"] for c in calls
    )


def test_selected_cli_dry_run_does_not_read_key_or_create_output(tmp_path):
    env = os.environ.copy()
    env.pop("DEEPSEEK_API_KEY", None)
    target = tmp_path / "no-run"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_memory_interactions.py",
            "--suite",
            "analyst-v1",
            "--analysis-mode",
            "analyst",
            "--output-dir",
            str(target),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)
    assert plan["max_model_calls"] == 40 and plan["api_calls"] == 0
    assert plan["suite"] == "analyst-v1" and plan["analysis_mode"] == "analyst"
    assert not target.exists()


@pytest.mark.parametrize("mode", ["direct", "analyst"])
def test_pre_call_rejection_is_not_counted_as_a_sent_request(tmp_path, mode):
    cases = load_interactions(FIXTURE)
    fixed = paired_client(cases)
    sent = []
    checks = []

    def check():
        checks.append(1)
        if len(checks) == 2:
            raise ProviderError("freeze_changed")

    def complete(messages):
        sent.append(messages)
        return fixed(messages)

    report = evaluate_interactions(
        cases,
        complete=complete,
        output_dir=tmp_path / "run",
        analysis_mode=mode,
        pre_call_check=check,
    )
    assert len(sent) == 1 and report["model_calls"] == 1
    assert report["preflight_errors"] == 1 and report["provider_errors"] == 0
    assert report["base_steps"] == 18 and report["confirmation_correct"] == 0
    rows = [json.loads(x) for x in (tmp_path / "run/steps.jsonl").read_text().splitlines()]
    assert any(r.get("error_code") == "freeze_changed" for r in rows)
