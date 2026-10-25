"""外部对话只经真实宿主提取/提交，评分参考不能补运行状态。"""

import json
from dataclasses import replace

import pytest
from test_external_derived import policy
from test_external_evaluation import fixture
from test_request_attributes import plain

from memrecagent.deepseek_client import ModelReply, ProviderError
from memrecagent.external_derived import derive_external_suite
from memrecagent.external_replay import evaluate_external_replay, graded_ranking_metrics


def suite():
    s, t = fixture()
    return derive_external_suite(s, t, policy())


def complete(messages):
    p = json.loads(messages[-1]["content"])
    assert "item_scores" not in p and "reference" not in p and "for_person" not in p
    row = next(r for r in p["turns"] if r["turn_id"] == p["latest_user_turn_id"])
    from test_memory_interaction import turn

    t = turn(int(row["turn_id"][1:]), row["content"])
    if t.turn_id == "T1":
        from test_attribute_language import durable

        raw = durable(t)
    else:
        raw = plain(t)
    return ModelReply(json.dumps(raw, ensure_ascii=False), "fixed", None, None, 0.0)


@pytest.mark.parametrize("usage", [{}, {"prompt_tokens": 1}])
def test_partial_usage_keeps_terminal_report_and_unknown_total(tmp_path, usage):
    def partial(messages):
        return replace(complete(messages), usage=usage)

    report = evaluate_external_replay(
        suite(), complete=partial, output_dir=tmp_path / "run", analysis_mode="direct"
    )
    assert (tmp_path / "run/report.json").is_file()
    assert report["analysis_calls"] == 2
    assert report["calls_without_total_tokens"] == 2
    assert report["calls_without_reported_usage"] == 0
    assert report["reported_tokens"] == 0  # 已报告合计，不代表实际总量为0。
    assert report["reported_usage_totals"] == ({"prompt_tokens": 2} if usage else {})


def test_actual_live_cli_summary_reports_missing_total_without_network(
    tmp_path, monkeypatch, capsys
):
    import importlib.util
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    from test_external_derived import cli as derive_cli

    assert derive_cli(tmp_path, "--derive").returncode == 0
    spec = importlib.util.spec_from_file_location(
        "replay_cli", Path("scripts/evaluate_external_replay.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "synthetic-placeholder")
    monkeypatch.setattr(
        module,
        "DeepSeekClient",
        lambda key: SimpleNamespace(
            complete=lambda messages: replace(complete(messages), usage={"prompt_tokens": 1})
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replay",
            "--suite",
            str(tmp_path / "output/derived-suite.json"),
            "--output-dir",
            str(tmp_path / "run"),
            "--analysis-mode",
            "direct",
            "--live",
            "--max-calls",
            "2",
        ],
    )
    assert module.main() == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["api_calls"] == 2  # CLI live模式计实际适配回调，此测试供应商固定。
    assert summary["calls_without_total_tokens"] == 2
    assert summary["reported_usage_totals"] == {"prompt_tokens": 2}


def test_actual_replay_from_empty_disk_uses_generated_budget_and_marks_algorithm_unavailable(
    tmp_path,
):
    result = evaluate_external_replay(
        suite(), complete=complete, output_dir=tmp_path / "run", analysis_mode="direct"
    )
    assert result["planned_steps"] == 2 and result["successful_steps"] == 2
    assert result["analysis_calls"] == 2 and result["api_calls"] == 0
    row = json.loads((tmp_path / "run/cases.jsonl").read_text())
    assert len(row["groups"]["G00"]["item_ids"]) == 2
    assert len(row["groups"]["G10"]["item_ids"]) == 1
    assert row["groups"]["G01"]["status"] == row["groups"]["G11"]["status"] == "ready"
    assert row["groups"]["G01"]["algorithm_effective"] is False
    assert row["groups"]["G11"]["used_strategy"] == "popularity"
    assert result["algorithm_effect"] is None and result["interaction_effect"] is None
    assert result["state_matches"] == 2


def test_processing_failure_is_not_repaired_with_gold_budget(tmp_path):
    def no_budget(messages):
        p = json.loads(messages[-1]["content"])
        if p["latest_user_turn_id"] == "T1":
            return ModelReply("{}", "fixed", None, None, 0.0)
        return complete(messages)

    result = evaluate_external_replay(
        suite(), complete=no_budget, output_dir=tmp_path / "run", analysis_mode="direct"
    )
    assert result["processing_errors"] == 1 and result["state_matches"] == 0
    row = json.loads((tmp_path / "run/cases.jsonl").read_text())
    assert len(row["groups"]["G10"]["item_ids"]) == 2


def test_provider_failure_stops_remaining_calls_and_keeps_case_denominator(tmp_path):
    sent = []

    def fail(messages):
        sent.append(1)
        raise ProviderError("http", 503)

    result = evaluate_external_replay(
        suite(), complete=fail, output_dir=tmp_path / "run", analysis_mode="direct"
    )
    assert len(sent) == 1 and result["planned_steps"] == 2 and result["not_run"] == 1
    assert result["planned_cases"] == 1 and result["case_errors"] == 1
    assert result["calls_without_reported_usage"] == 1


def test_budget_never_starts_partial_dual_analysis_turn(tmp_path):
    sent = []

    def counted(messages):
        sent.append(1)
        return complete(messages)

    result = evaluate_external_replay(
        suite(), complete=counted, output_dir=tmp_path / "run", analysis_mode="analyst", max_calls=3
    )
    assert len(sent) == 2 and result["not_run"] == 1 and result["planned_steps"] == 2


def test_graded_gain_masks_ineligible_and_empty_failure_is_zero_not_dropped():
    ref = [
        dict(item_id="A", relevance=3, eligible=False),
        dict(item_id="B", relevance=2, eligible=True),
    ]
    bad = graded_ranking_metrics(["A"], ref, k=1)
    good = graded_ranking_metrics(["B"], ref, k=1)
    empty = graded_ranking_metrics([], ref, k=1)
    assert bad["ndcg"] == empty["ndcg"] == 0 and good["ndcg"] == 1
    assert bad["constraint_violations"] == 1 and good["constraint_violations"] == 0


def test_no_positive_reference_keeps_undefined_metrics_unknown():
    scores = graded_ranking_metrics([], [dict(item_id="A", relevance=0, eligible=True)], k=1)
    assert scores["ndcg"] is scores["hit_rate"] is scores["mrr"] is scores["precision"] is None


def test_changed_frozen_input_stops_remaining_attempts(tmp_path):
    sent = []

    def changed(messages):
        sent.append(1)
        reply = complete(messages)
        (tmp_path / "run/inputs.json").write_text("{}")
        return reply

    result = evaluate_external_replay(
        suite(), complete=changed, output_dir=tmp_path / "run", analysis_mode="direct"
    )
    assert len(sent) == 1 and result["not_run"] == 1 and not result["inputs_source_unchanged"]


def test_corrupt_runtime_store_leaves_terminal_report_and_all_case_denominators(tmp_path):
    from hashlib import sha256

    def corrupt(messages):
        reply = complete(messages)
        store = tmp_path / "run/stores/X"
        store.mkdir(parents=True, exist_ok=True)
        (store / (sha256(b"synthetic-external:X").hexdigest() + ".jsonl")).write_text(
            "invalid-runtime-store\n"
        )
        return reply

    result = evaluate_external_replay(
        suite(), complete=corrupt, output_dir=tmp_path / "run", analysis_mode="direct"
    )
    assert result["planned_steps"] == 2 and result["not_run"] == 1
    assert result["planned_cases"] == 1 and result["case_errors"] == 1


def test_added_source_inventory_entry_stops_before_second_call(tmp_path, monkeypatch):
    import memrecagent.external_replay as evaluation

    original = evaluation._source_paths
    sent = []
    extra = tmp_path / "injected.py"
    extra.write_text("# fixture\n")

    def paths():
        actual = original()
        if sent:
            actual["extra-source.py"] = extra
        return actual

    def changed(messages):
        sent.append(1)
        return complete(messages)

    monkeypatch.setattr(evaluation, "_source_paths", paths)
    report = evaluation.evaluate_external_replay(
        suite(), complete=changed, output_dir=tmp_path / "run", analysis_mode="direct"
    )
    assert len(sent) == 1 and report["not_run"] == 1 and not report["inputs_source_unchanged"]


def test_cli_defaults_to_no_api_plan_and_requires_verified_derived_bytes_and_live_key(tmp_path):
    import os
    import subprocess
    import sys

    from test_external_derived import cli as derive_cli

    assert derive_cli(tmp_path, "--derive").returncode == 0
    path = tmp_path / "output/derived-suite.json"
    command = [
        sys.executable,
        "scripts/evaluate_external_replay.py",
        "--suite",
        str(path),
        "--output-dir",
        str(tmp_path / "replay"),
    ]
    env = {k: v for k, v in os.environ.items() if k != "DEEPSEEK_API_KEY"}
    plan = subprocess.run(command, capture_output=True, env=env, timeout=15)
    assert plan.returncode == 0, plan.stderr
    assert json.loads(plan.stdout)["api_calls"] == 0 and not (tmp_path / "replay").exists()
    live = subprocess.run(
        command + ["--live", "--max-calls", "4"], capture_output=True, env=env, timeout=15
    )
    assert live.returncode == 2 and not (tmp_path / "replay").exists()
    assert json.loads(live.stderr) == {"error": "invalid_external_replay_input"}
    path.write_bytes(path.read_bytes() + b"\n")
    assert subprocess.run(command, capture_output=True, env=env, timeout=15).returncode == 2


def test_effect_interval_resamples_whole_families_not_individual_near_templates():
    from memrecagent.external_replay import family_memory_contrast

    rows = []
    for family, off, on in [("a", 0, 1), ("a", 0, 1), ("b", 1, 0)]:
        rows.append(
            dict(
                template_family=family,
                reference_status="provisional",
                groups={"G00": dict(metrics=dict(ndcg=off)), "G10": dict(metrics=dict(ndcg=on))},
            )
        )
    report = family_memory_contrast(rows, seed=9)
    assert report["n_families"] == 2 and report["n_cases"] == 3
    assert report["delta"] == 0
    assert report["interval"] == [-1, 1]


def test_episode_identifier_cannot_escape_runtime_store_directory(tmp_path):
    import pytest

    data = suite()
    data["episodes"][0]["episode_id"] = "../../escaped"
    data["tasks"][0]["episode_id"] = "../../escaped"
    with pytest.raises(ValueError):
        evaluate_external_replay(
            data, complete=complete, output_dir=tmp_path / "run", analysis_mode="direct"
        )
    assert not (tmp_path / "escaped").exists()


def test_negative_seed_is_rejected_before_consuming_calls_or_creating_output(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        evaluate_external_replay(
            suite(), complete=complete, output_dir=tmp_path / "run", analysis_mode="direct", seed=-1
        )
    assert not (tmp_path / "run").exists()
