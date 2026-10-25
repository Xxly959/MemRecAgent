"""同一实际前文的两个明确LLM系统基线，不以固定客户端计模型成绩。"""

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from memrecagent.deepseek_client import ModelReply, ProviderError
from memrecagent.llm_recommendation_baselines import (
    baseline_messages,
    decode_baseline_ranking,
    evaluate_baselines,
    parse_baseline_suite,
    select_history,
)


def suite():
    runtime = dict(
        query=dict(turn_id="T6", session_id="S2", content="推荐手机", offset_seconds=60),
        k=3,
        history=[
            dict(
                turn_id=f"T{i}",
                session_id="S1",
                role="user",
                content="喜欢ＳＡＭＳＵＮＧ" if i == 1 else f"闲聊{i}",
                offset_seconds=i * 10,
            )
            for i in range(1, 6)
        ],
        items=[
            dict(item_id="B", brand="品牌B", category="phone", price_cny=2500, public_prior=10),
            dict(item_id="C", brand="品牌C", category="phone", price_cny=2000, public_prior=8),
        ],
    )
    return dict(
        schema_version=1,
        synthetic=True,
        label_source="known unit fixture",
        cases=[
            dict(
                case_id=cid,
                cluster_id=cid,
                split="dev",
                runtime=deepcopy(runtime),
                reference=dict(
                    status="provisional",
                    item_scores=[
                        dict(item_id="B", eligible=True, relevance=1),
                        dict(item_id="C", eligible=True, relevance=3),
                    ],
                ),
                anchor=dict(
                    status="ready",
                    item_ids=["C", "B"],
                    analysis_calls=6,
                    reported_tokens=60,
                    calls_without_usage=0,
                ),
            )
            for cid in ("a", "b")
        ],
    )


def reply(content='{"item_ids":["C","B"]}', *, usage=True):
    return ModelReply(
        content,
        "fixed",
        None,
        dict(total_tokens=10, prompt_tokens=8, completion_tokens=2) if usage else None,
        0,
    )


def complete(messages):
    payload = json.loads(messages[-1]["content"])
    assert set(payload) == {"query", "history", "items", "k"}
    assert "item_scores" not in messages[-1]["content"] and "anchor" not in messages[-1]["content"]
    return reply(
        json.dumps(dict(item_ids=["B", "C"] if len(payload["history"]) > 3 else ["C", "B"]))
    )


def test_character_tfidf_matches_nfkc_case_and_zero_overlap_prefers_recent():
    runtime = suite()["cases"][0]["runtime"]
    runtime["query"]["content"] = "samsung"
    assert [
        t["turn_id"] for t in select_history(runtime, "char_tfidf_user_turns_llm", top_k=1)
    ] == ["T1"]
    runtime["query"]["content"] = "完全陌生的词"
    assert [
        t["turn_id"] for t in select_history(runtime, "char_tfidf_user_turns_llm", top_k=2)
    ] == ["T4", "T5"]
    assert len(select_history(runtime, "raw_history_llm")) == 5


def test_assistant_is_context_not_indexed_as_user_preference():
    runtime = suite()["cases"][0]["runtime"]
    runtime["history"].insert(
        4,
        dict(
            turn_id="A4",
            session_id="S1",
            role="assistant",
            content="samsung很好，要确认吗",
            offset_seconds=42,
        ),
    )
    runtime["query"]["content"] = "闲聊5"
    selected = select_history(runtime, "char_tfidf_user_turns_llm", top_k=1)
    assert [t["turn_id"] for t in selected] == ["A4", "T5"]
    assert selected[0]["role"] == "assistant" and selected[1]["role"] == "user"


def test_messages_share_system_and_catalog_reference_changes_never_change_payload():
    s = suite()
    runtime = s["cases"][0]["runtime"]
    raw = baseline_messages(runtime, "raw_history_llm")
    retrieved = baseline_messages(runtime, "char_tfidf_user_turns_llm")
    assert raw[0] == retrieved[0]
    p, q = (json.loads(x[-1]["content"]) for x in (raw, retrieved))
    assert p["items"] == q["items"] and p["query"] == q["query"] and p["k"] == q["k"]
    s["cases"][0]["reference"]["item_scores"][0]["relevance"] = 3
    assert baseline_messages(parse_baseline_suite(s)[0]["runtime"], "raw_history_llm") == raw


@pytest.mark.parametrize(
    "bad",
    [
        "future",
        "query_in_history",
        "duplicate_id",
        "unknown_role",
        "negative_prior",
        "split_family",
        "unknown_reference_id",
    ],
)
def test_invalid_runtime_rejected_before_output_or_model(tmp_path, bad):
    s = suite()
    runtime = s["cases"][0]["runtime"]
    if bad == "future":
        runtime["history"][-1]["offset_seconds"] = 61
    elif bad == "query_in_history":
        runtime["history"][-1]["turn_id"] = "T6"
    elif bad == "duplicate_id":
        runtime["items"][1]["item_id"] = "B"
    elif bad == "unknown_role":
        runtime["history"][0]["role"] = "tool"
    elif bad == "negative_prior":
        runtime["items"][0]["public_prior"] = -1
    elif bad == "split_family":
        s["cases"][1].update(cluster_id="a", split="test")
    else:
        s["cases"][0]["reference"]["item_scores"][0]["item_id"] = "unknown"
    with pytest.raises(ValueError):
        evaluate_baselines(
            s, complete=lambda _: pytest.fail("must not call"), output_dir=tmp_path / "bad"
        )
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    "raw",
    [
        '{"item_ids":["unknown"]}',
        '{"item_ids":["C","C"]}',
        '{"item_ids":[true]}',
        '{"item_ids":[],"reply":"x"}',
        '{"item_ids":[],"item_ids":[]}',
        '```json\n{"item_ids":[]}\n```',
    ],
)
def test_model_ranking_requires_unique_bounded_catalog_ids(raw):
    with pytest.raises(ValueError):
        decode_baseline_ranking(raw, {"B", "C"}, k=3)


def test_real_runner_pairs_arms_counts_cost_and_compares_equal_family_scores(tmp_path):
    report = evaluate_baselines(suite(), complete=complete, output_dir=tmp_path / "run")
    assert report["client_attempts"] == 4 and report["api_calls"] == 0
    assert report["reported_tokens"] == 40 and report["unknown_usage_calls"] == 0
    assert report["comparison_complete"] and report["inputs_source_unchanged"]
    primary = report["summaries"]["provisional"]
    assert primary["arms"]["raw_history_llm"]["ndcg"] == pytest.approx(0.7098097413968655)
    assert primary["arms"]["char_tfidf_user_turns_llm"]["ndcg"] == 1
    assert primary["contrasts"]["retrieval_minus_raw"]["delta"] == pytest.approx(0.2901902586031345)
    rows = [json.loads(line) for line in (tmp_path / "run/calls.jsonl").read_text().splitlines()]
    assert rows[0]["arm"] != rows[2]["arm"]  # 两案例交替先手
    before = (tmp_path / "run/report.json").read_bytes()
    with pytest.raises(FileExistsError):
        evaluate_baselines(suite(), complete=complete, output_dir=tmp_path / "run")
    assert before == (tmp_path / "run/report.json").read_bytes()


def test_parse_failure_continues_other_arm_but_provider_failure_stops_with_missing_pairs(tmp_path):
    calls = []

    def malformed_first(messages):
        calls.append(messages)
        return reply("{}") if len(calls) == 1 else complete(messages)

    r = evaluate_baselines(suite(), complete=malformed_first, output_dir=tmp_path / "parse")
    assert r["client_attempts"] == 4 and r["comparison_complete"]
    assert sum(a["n_errors"] for a in r["summaries"]["provisional"]["arms"].values()) == 1

    def fail(_):
        raise ProviderError("http", 503)

    r = evaluate_baselines(suite(), complete=fail, output_dir=tmp_path / "provider")
    assert (
        r["client_attempts"] == 1 and r["unknown_usage_calls"] == 1 and not r["comparison_complete"]
    )
    assert r["not_run_arms"] == 3
    assert r["summaries"]["provisional"]["contrasts"]["retrieval_minus_raw"]["interval"] is None


def test_full_pair_budget_and_metadata_failure_do_not_claim_zero_cost(tmp_path):
    r = evaluate_baselines(suite(), complete=complete, output_dir=tmp_path / "budget", max_calls=3)
    assert r["client_attempts"] == 2 and r["not_run_arms"] == 2 and not r["comparison_complete"]
    bad = reply()
    object.__setattr__(bad, "latency_seconds", float("nan"))
    r = evaluate_baselines(suite(), complete=lambda _: bad, output_dir=tmp_path / "metadata")
    assert (
        r["client_attempts"] == 1 and r["unknown_usage_calls"] == 1 and not r["comparison_complete"]
    )


def test_cli_plan_and_missing_live_key_do_not_create_outputs(tmp_path):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(suite()))
    out = tmp_path / "output"
    root = Path(__file__).resolve().parents[1]
    args = [
        sys.executable,
        str(root / "scripts/evaluate_llm_baselines.py"),
        "--suite",
        str(path),
        "--output-dir",
        str(out),
    ]
    env = {k: v for k, v in os.environ.items() if k != "DEEPSEEK_API_KEY"}
    p = subprocess.run(args, capture_output=True, text=True, env=env)
    assert p.returncode == 0 and json.loads(p.stdout)["planned_max_calls"] == 4 and not out.exists()
    p = subprocess.run(
        args + ["--live", "--max-calls", "4"], capture_output=True, text=True, env=env
    )
    assert p.returncode == 2 and p.stderr == "baseline_unavailable\n" and not out.exists()


def test_input_changed_while_building_messages_is_not_sent(tmp_path, monkeypatch):
    import memrecagent.llm_recommendation_baselines as module

    data = suite()
    path = tmp_path / "input.json"
    path.write_text(json.dumps(data))
    original = module.baseline_messages

    def changed(runtime, arm):
        messages = original(runtime, arm)
        path.write_text("changed after preflight")
        return messages

    monkeypatch.setattr(module, "baseline_messages", changed)
    calls = []

    def complete(messages):
        calls.append(messages)
        return reply()

    result = evaluate_baselines(
        data, complete=complete, output_dir=tmp_path / "run", input_file=path
    )
    assert not calls
    assert result["client_attempts"] == 0 and result["not_run_arms"] == 4
    assert not result["inputs_source_unchanged"] and not result["comparison_complete"]
