"""等调用上限的草案可见性控制；固定响应不计模型准确率。"""

import json
import os
import subprocess
import sys
from dataclasses import replace
from hashlib import sha256

import pytest
from test_external_replay import complete as external_complete
from test_external_replay import suite as external_suite
from test_memory_analyst import reply, task_for
from test_memory_analyst_host import FixedStages, old, setup
from test_memory_interaction import after, turn, value
from test_memory_quality import client, suite

from memrecagent.deepseek_client import ProviderError
from memrecagent.external_replay import evaluate_external_replay
from memrecagent.memory_analyst import AnalystError, build_analyst_messages
from memrecagent.memory_quality_evaluation import evaluate_memory_quality


def test_only_draft_visibility_changes_with_same_context_role_and_view(tmp_path):
    t = turn(1, "我现在取消对小米的排除。")
    draft = value(t, entity="小米", relation="prefer")
    final = value(t, entity="小米", relation="neutral")
    messages_by_mode = {}
    for mode in ("analyst", "analyst_blind"):
        store, h = setup(tmp_path / mode, (old(),))
        stages = FixedStages(draft, final)
        sent = []

        def complete(messages, *, captured=sent, current_store=store, client=stages):
            captured.append(json.loads(json.dumps(messages)))
            assert [m.relation for m in current_store.list_memories("u", as_of=after(t))] == [
                "exclude"
            ]
            return client(messages)

        preview = h.prepare_user(t, complete=complete, as_of=after(t), analysis_mode=mode)
        assert len(sent) == 2
        assert [c.stage for c in preview.analysis_calls] == ["draft", "analyst"]
        h.commit(preview.ticket_id, as_of=after(t))
        assert [m.relation for m in store.list_memories("u", as_of=after(t))] == [
            "exclude",
            "neutral",
        ]
        messages_by_mode[mode] = sent
    normal, blind = messages_by_mode["analyst"], messages_by_mode["analyst_blind"]
    assert normal[0] == blind[0]
    assert normal[1][0] == blind[1][0]
    normal_body = json.loads(normal[1][1]["content"])
    blind_body = json.loads(blind[1][1]["content"])
    assert normal_body.pop("draft") == draft
    assert blind_body.pop("draft") is None
    assert normal_body == blind_body
    assert [(e["entity"], e["relation"]) for e in blind_body["memory_view"]["entries"]] == [
        ("小米", "exclude")
    ]


def test_invalid_draft_text_and_error_are_hidden_but_attempt_is_recorded(tmp_path):
    store, h = setup(tmp_path, (old(),))
    t = turn(1, "我以后优先小米手机。")
    stages = FixedStages(
        reply({}, content='{"INTERNAL_DRAFT_SENTINEL":true}'),
        value(t, entity="小米", relation="prefer"),
    )
    preview = h.prepare_user(t, complete=stages, as_of=after(t), analysis_mode="analyst_blind")
    assert len(stages.seen) == 2
    assert stages.seen[1]["draft"] is None
    assert "untrusted_draft" not in stages.seen[1]
    assert "INTERNAL_DRAFT_SENTINEL" not in json.dumps(stages.seen[1])
    assert stages.seen[1]["memory_view"]["entries"] == []
    assert [c.status for c in preview.analysis_calls] == ["processing_error", "ok"]
    assert [m.relation for m in store.list_memories("u", as_of=after(t))] == ["exclude"]


@pytest.mark.parametrize("bad", [0, 1, None, "false"])
def test_draft_visibility_is_strict_trusted_boolean(bad):
    t = turn(1, "我喜欢小米。")
    with pytest.raises(AnalystError, match="task"):
        build_analyst_messages(replace(task_for(t), expose_draft=bad))


@pytest.mark.parametrize("stage", [1, 2])
def test_blind_provider_failure_stops_without_retries_or_draft_fallback(tmp_path, stage):
    store, h = setup(tmp_path)
    t = turn(1, "我一直喜欢小米。")
    responses = (
        (ProviderError("transport"),)
        if stage == 1
        else (value(t, entity="小米", relation="prefer"), ProviderError("transport"))
    )
    stages = FixedStages(*responses)
    with pytest.raises(AnalystError, match="transport"):
        h.prepare_user(t, complete=stages, as_of=after(t), analysis_mode="analyst_blind")
    assert len(stages.seen) == stage
    assert len(h.last_analysis_calls) == stage
    assert not store.list_memories("u", as_of=after(t))


def test_quality_runner_accounts_for_both_blind_calls(tmp_path):
    seen = []
    report = evaluate_memory_quality(
        suite(),
        complete=client(seen=seen),
        output_dir=tmp_path / "run",
        analysis_mode="analyst_blind",
        max_calls=4,
    )
    assert report["provider_attempts"] == 4
    assert report["strict_steps"] == 2
    assert len(seen) == 4
    assert all(p["draft"] is None and "untrusted_draft" not in p for p in seen[1::2])


def test_external_runner_reserves_full_blind_turn_budget_and_labels_stages(tmp_path):
    sent = []

    def counted(messages):
        sent.append(json.loads(messages[-1]["content"]))
        return external_complete(messages)

    report = evaluate_external_replay(
        external_suite(),
        complete=counted,
        output_dir=tmp_path / "run",
        analysis_mode="analyst_blind",
        max_calls=3,
    )
    assert len(sent) == report["analysis_calls"] == 2
    assert sent[1]["draft"] is None
    assert report["not_run"] == 1
    calls = [json.loads(line) for line in (tmp_path / "run/calls.jsonl").read_text().splitlines()]
    assert [c["stage"] for c in calls] == ["draft", "analyst"]


@pytest.mark.parametrize("kind", ["quality", "external"])
def test_blind_cli_plan_without_key_uses_two_call_upper_bound(tmp_path, kind):
    env = os.environ.copy()
    env.pop("DEEPSEEK_API_KEY", None)
    command = [sys.executable, "scripts/evaluate_memory_quality.py"]
    if kind == "external":
        raw = json.dumps(external_suite()).encode()
        (tmp_path / "suite.json").write_bytes(raw)
        (tmp_path / "manifest.json").write_text(
            json.dumps(dict(derived_sha256=sha256(raw).hexdigest()))
        )
        command = [
            sys.executable,
            "scripts/evaluate_external_replay.py",
            "--suite",
            str(tmp_path / "suite.json"),
        ]
    command += ["--output-dir", str(tmp_path / "run"), "--analysis-mode", "analyst_blind"]
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["api_calls"] == 0
    if kind == "quality":
        assert plan["max_provider_attempts"] == 192 and plan["n_steps"] == 96
    else:
        assert plan["planned_max_calls"] == 4 and plan["planned_steps"] == 2
    assert not (tmp_path / "run").exists()
