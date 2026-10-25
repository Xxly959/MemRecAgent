"""共享配对评分及物理成本必须来自实际回调，失败不从分母剔除。"""

import json
import os
import subprocess
import sys
from copy import deepcopy

import pytest
from test_memory_analyst import reply
from test_memory_interaction import after, turn, value

from memrecagent.deepseek_client import ProviderError


def suite():
    t = turn(1, "我以后一直喜欢三星手机。")
    runtime = dict(
        turns=[
            dict(
                turn_id=t.turn_id,
                role=t.role,
                content=t.content,
                occurred_at=t.occurred_at.isoformat(),
                available_at=t.available_at.isoformat(),
            )
        ],
        as_of=after(t).isoformat(),
        active_memories=[],
        pending_question_turn_id=None,
    )
    return dict(
        schema_version=1,
        synthetic=True,
        label_source="author-known synthetic unit fixture",
        cases=[
            dict(
                case_id=cid,
                cluster_id=cid,
                split="dev",
                runtime=deepcopy(runtime),
                references=[value(t, entity="三星", relation="prefer")],
            )
            for cid in ("a", "b")
        ],
    )


def complete(messages):
    payload = json.loads(messages[-1]["content"])
    assert "label_source" not in payload and "cases" not in payload and "references" not in payload
    t = turn(1, payload["turns"][-1]["content"])
    relation = "dislike" if payload.get("draft") is not None else "prefer"
    return reply(value(t, entity="三星", relation=relation))


def evaluate(data, tmp_path, **kwargs):
    from memrecagent.shared_analysis_evaluation import evaluate_shared_analysis

    return evaluate_shared_analysis(
        data, complete=kwargs.pop("complete", complete), output_dir=tmp_path / "run", **kwargs
    )


def test_actual_shared_callbacks_are_counted_once_and_paired_by_case(tmp_path):
    report = evaluate(suite(), tmp_path)
    assert report["client_attempts"] == report["callback_attempts"] == 6
    assert report["api_calls"] == 0 and report["reported_total_tokens"] == 30
    assert report["comparison_ready"]
    assert report["arms"]["direct"]["matches"] == 2
    assert report["arms"]["analyst"]["matches"] == 0
    assert report["arms"]["analyst_blind"]["matches"] == 2
    contrast = report["contrasts"]["analyst_blind_minus_analyst"]
    assert contrast["delta"] == 1 and contrast["interval"] == [1, 1]
    calls = [json.loads(line) for line in (tmp_path / "run/calls.jsonl").read_text().splitlines()]
    assert sum(row["stage"] == "draft" for row in calls) == 2
    assert all(row["sent"] for row in calls)
    assert report["inputs_source_unchanged"]


def test_budget_keeps_unstarted_pair_in_denominator_without_half_pair(tmp_path):
    report = evaluate(suite(), tmp_path, max_calls=5)
    assert report["client_attempts"] == 3 and report["planned_cases"] == 2
    assert report["arms"]["analyst_blind"]["matches"] == 1
    assert report["arms"]["analyst_blind"]["not_run"] == 1
    assert not report["comparison_ready"]
    assert report["contrasts"]["analyst_blind_minus_analyst"]["interval"] is None


def test_provider_failure_stops_remaining_cases_and_preserves_unknown_cost(tmp_path):
    def fail(_):
        raise ProviderError("http", 503)

    report = evaluate(suite(), tmp_path, complete=fail)
    assert report["client_attempts"] == 1 and report["calls_without_total_tokens"] == 1
    assert report["arms"]["direct"]["matches"] == 0
    assert report["arms"]["analyst_blind"]["not_run"] == 2
    assert not report["comparison_ready"]


@pytest.mark.parametrize("kind", ["reference", "cross_split", "duplicate"])
def test_invalid_suite_is_rejected_before_output_or_callbacks(tmp_path, kind):
    data = suite()
    if kind == "reference":
        data["cases"][1]["references"] = [{"forged": True}]
    elif kind == "cross_split":
        data["cases"][1].update(cluster_id="a", split="test")
    else:
        data["cases"][1]["case_id"] = "a"

    def never(_):
        pytest.fail("无效数据不得调用模型")

    with pytest.raises(ValueError):
        evaluate(data, tmp_path, complete=never)
    assert not (tmp_path / "run").exists()


def test_input_file_change_stops_after_first_return_without_retries(tmp_path):
    source = tmp_path / "input.json"
    source.write_text(json.dumps(suite()))

    def changed(messages):
        source.write_text(source.read_text() + "\n")
        return complete(messages)

    report = evaluate(suite(), tmp_path, complete=changed, input_file=source)
    assert report["client_attempts"] == 1 and not report["inputs_source_unchanged"]
    assert not report["comparison_ready"]


def test_cli_plan_is_zero_call_and_live_needs_explicit_key_and_budget(tmp_path):
    source = tmp_path / "input.json"
    source.write_text(json.dumps(suite()))
    command = [
        sys.executable,
        "scripts/evaluate_shared_analysis.py",
        "--suite",
        str(source),
        "--output-dir",
        str(tmp_path / "run"),
    ]
    env = {k: v for k, v in os.environ.items() if k != "DEEPSEEK_API_KEY"}
    plan = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
    assert plan.returncode == 0, plan.stderr
    assert json.loads(plan.stdout)["planned_max_calls"] == 6
    assert json.loads(plan.stdout)["api_calls"] == 0
    live = subprocess.run(
        command + ["--live", "--max-calls", "6"],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert live.returncode == 2
    assert not (tmp_path / "run").exists()


def test_contrast_weights_whole_clusters_instead_of_duplicate_case_counts(tmp_path):
    data = suite()
    original = data["cases"][0]
    data["cases"] = [
        dict(deepcopy(original), case_id="a" + str(i), cluster_id="a") for i in range(3)
    ]
    b = deepcopy(original)
    b.update(case_id="b", cluster_id="b")
    t = turn(1, "我以后一直喜欢华为手机。")
    b["runtime"]["turns"][0]["content"] = t.content
    b["references"] = [value(t, entity="华为", relation="prefer")]
    data["cases"].append(b)

    def polarity(messages):
        p = json.loads(messages[-1]["content"])
        t = turn(1, p["turns"][-1]["content"])
        entity = "华为" if "华为" in t.content else "三星"
        normal = p.get("draft") is not None
        wrong = (entity == "三星" and normal) or (entity == "华为" and "draft" in p and not normal)
        return reply(value(t, entity=entity, relation="dislike" if wrong else "prefer"))

    r = evaluate(data, tmp_path, complete=polarity)
    contrast = r["contrasts"]["analyst_blind_minus_analyst"]
    assert contrast["delta"] == 0 and contrast["interval"] == [-1, 1]
    assert contrast["n_clusters"] == 2 and contrast["n_cases"] == 4


@pytest.mark.parametrize("kind", ["deep_json", "seed", "budget"])
def test_cli_rejects_bad_inputs_with_fixed_error_and_no_traceback(tmp_path, kind):
    source = tmp_path / "input.json"
    source.write_text("[" * 20000 + "]" * 20000 if kind == "deep_json" else json.dumps(suite()))
    command = [
        sys.executable,
        "scripts/evaluate_shared_analysis.py",
        "--suite",
        str(source),
        "--output-dir",
        str(tmp_path / "run"),
    ]
    if kind == "seed":
        command += ["--seed", "-1"]
    if kind == "budget":
        command += ["--max-calls", "1"]
    r = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert r.returncode == 2
    assert json.loads(r.stderr) == {"error": "invalid_shared_analysis_input"}
    assert not (tmp_path / "run").exists()
