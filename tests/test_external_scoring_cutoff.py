"""新评分协议事前绑定请求数量；不取模型返回数或答案决定分母。"""

import json

import pytest
from test_external_replay import complete, suite

from memrecagent.external_replay import evaluate_external_replay


def count_suite(text="推荐一款手机。"):
    data = suite()
    data["episodes"][0]["steps"][-1]["text"] = text
    data["tasks"][0]["runtime"]["request_text"] = text
    return data


def run(data, path, **kwargs):
    return evaluate_external_replay(
        data, complete=complete, output_dir=path, analysis_mode="direct", **kwargs
    )


def case(path):
    return json.loads((path / "cases.jsonl").read_text())


def test_request_cutoff_uses_one_product_for_all_arms_without_changing_calls_or_ranks(tmp_path):
    data = count_suite()
    # 两个商品都满足预算且同样相关，最佳单件@1为1，@3不能达到1。
    for item in data["tasks"][0]["runtime"]["items"]:
        item["price_cny"] = 2500
    for item in data["tasks"][0]["reference"]["item_scores"]:
        item.update(eligible=True, relevance=3)
    old = run(data, tmp_path / "fixed")
    new = run(data, tmp_path / "request", scoring_mode="request")
    fixed_case, bound_case = case(tmp_path / "fixed"), case(tmp_path / "request")
    assert new["analysis_calls"] == old["analysis_calls"] == 2
    assert bound_case["scoring_cutoff"] == {
        "k": 1,
        "source": "explicit_request",
        "evidence_spans": [[2, 4]],
    }
    assert fixed_case["scoring_cutoff"]["k"] == 3
    for name in fixed_case["groups"]:
        assert fixed_case["groups"][name]["item_ids"] == bound_case["groups"][name]["item_ids"]
    assert bound_case["groups"]["G10"]["metrics"]["ndcg"] == 1
    assert bound_case["groups"]["G10"]["metrics"]["precision"] == 1
    assert fixed_case["groups"]["G10"]["metrics"]["ndcg"] < 1
    assert new["scoring_k_distribution"] == {"1": 1}
    assert new["scoring_protocol"] == "request_count_v1"
    assert old["scoring_protocol"] == "fixed_k_v1"

    def messages(folder):
        return [
            json.loads(line)["messages"]
            for line in (folder / "calls.jsonl").read_text().splitlines()
        ]

    assert messages(tmp_path / "fixed") == messages(tmp_path / "request")


def test_request_cutoff_is_fixed_even_when_runtime_returns_fewer_items_or_errors(tmp_path):
    data = count_suite("推荐两款手机。")
    report = run(data, tmp_path / "request", scoring_mode="request")
    row = case(tmp_path / "request")
    assert row["scoring_cutoff"]["k"] == 2
    assert len(row["groups"]["G10"]["item_ids"]) == 1
    assert row["groups"]["G10"]["metrics"]["precision"] == 0.5
    assert report["scoring_k_distribution"] == {"2": 1}


def test_query_without_explicit_count_uses_given_fallback(tmp_path):
    run(count_suite("推荐手机。"), tmp_path / "run", scoring_mode="request", k=2)
    assert case(tmp_path / "run")["scoring_cutoff"] == {
        "k": 2,
        "source": "fallback",
        "evidence_spans": [],
    }


@pytest.mark.parametrize("text", ["推荐两到三款手机。", "推荐两款，再给三款。", "推荐十一款手机。"])
def test_ambiguous_request_is_rejected_before_model_or_output_creation(tmp_path, text):
    sent = []
    with pytest.raises(ValueError, match="scoring"):
        evaluate_external_replay(
            count_suite(text),
            complete=lambda messages: sent.append(messages),
            output_dir=tmp_path / "run",
            scoring_mode="request",
        )
    assert sent == [] and not (tmp_path / "run").exists()


@pytest.mark.parametrize("mode", [None, True, "returned_count"])
def test_unknown_scoring_mode_is_rejected_before_output(tmp_path, mode):
    with pytest.raises(ValueError):
        run(count_suite(), tmp_path / "run", scoring_mode=mode)
    assert not (tmp_path / "run").exists()


def test_provider_failure_keeps_planned_request_denominator(tmp_path):
    from memrecagent.deepseek_client import ProviderError

    def fail(messages):
        raise ProviderError("transport")

    report = evaluate_external_replay(
        count_suite("推荐两款手机。"),
        complete=fail,
        output_dir=tmp_path / "run",
        analysis_mode="direct",
        scoring_mode="request",
    )
    row = case(tmp_path / "run")
    assert row["scoring_cutoff"]["k"] == 2
    assert all(
        g["metrics"]["ndcg"] == g["metrics"]["precision"] == 0 for g in row["groups"].values()
    )
    assert report["provider_errors"] == 1 and report["not_run"] == 1


def test_reference_count_cannot_override_original_request(tmp_path):
    data = count_suite()
    data["tasks"][0]["reference"]["requested_count"] = 10
    run(data, tmp_path / "run", scoring_mode="request")
    assert case(tmp_path / "run")["scoring_cutoff"]["k"] == 1


def test_cli_request_plan_is_zero_call_and_rejects_ambiguous_input(tmp_path):
    import hashlib
    import os
    import subprocess
    import sys

    data = count_suite()
    path = tmp_path / "suite.json"

    def save():
        raw = json.dumps(data, ensure_ascii=False).encode()
        path.write_bytes(raw)
        (tmp_path / "manifest.json").write_text(
            json.dumps({"derived_sha256": hashlib.sha256(raw).hexdigest()})
        )

    save()
    args = [
        sys.executable,
        "scripts/evaluate_external_replay.py",
        "--suite",
        str(path),
        "--output-dir",
        str(tmp_path / "run"),
        "--scoring-mode",
        "request",
    ]
    env = {k: v for k, v in os.environ.items() if k != "DEEPSEEK_API_KEY"}
    result = subprocess.run(args, capture_output=True, text=True, env=env, timeout=15)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["api_calls"] == 0 and plan["scoring_cutoffs"]["X"]["k"] == 1
    assert plan["scoring_mode"] == "request"
    assert not (tmp_path / "run").exists()
    data = count_suite("推荐两到三款手机。")
    save()
    result = subprocess.run(args, capture_output=True, text=True, env=env, timeout=15)
    assert result.returncode == 2 and not (tmp_path / "run").exists()
