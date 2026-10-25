"""假模型只验证评测记账和副作用，不用于声称自然语言效果。"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_memory_extraction import output_for, turns_for

from memrecagent.deepseek_client import ModelReply, ProviderError
from memrecagent.memory_extraction_evaluation import build_messages, evaluate_cases, load_cases

FIXTURE = Path(__file__).parent / "fixtures/memory_extraction_cases.json"


def reply(raw):
    return ModelReply(raw, "deepseek-flash", "fp-synthetic", {"total_tokens": 12}, 0.1)


def test_correct_wrong_and_invalid_results_all_keep_their_denominator(tmp_path):
    cases = load_cases(FIXTURE)
    chosen = (cases[1], cases[0], cases[2])
    correct = output_for(turns_for(chosen[0]["turns"][0])[0])
    wrong = output_for(turns_for(chosen[1]["turns"][0])[0])
    wrong["assertions"][0].update(entity="三星", relation="dislike")
    responses = iter([reply(json.dumps(correct)), reply(json.dumps(wrong)), reply("{")])
    report = evaluate_cases(chosen, complete=lambda _: next(responses), output_dir=tmp_path / "run")
    assert report["total"] == report["attempted"] == 3
    assert report["strict_correct"] == 1
    assert report["extraction_failures"] == 1
    assert report["wrong_persistence_cases"] == 1
    assert report["usage_available_calls"] == 3 and report["tokens"]["total_tokens"] == 36
    rows = [json.loads(line) for line in (tmp_path / "run/cases.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and rows[2]["status"] == "extraction_error"
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["cases_sha256"] and manifest["prompt_sha256"]
    assert not (tmp_path / "run/memory").exists()


def test_provider_error_stops_remaining_calls_but_preserves_denominator(tmp_path):
    calls = []

    def fail(messages):
        calls.append(messages)
        raise ProviderError("http", 401)

    report = evaluate_cases(load_cases(FIXTURE), complete=fail, output_dir=tmp_path / "run")
    assert len(calls) == report["attempted"] == 1
    assert report["total"] == 24 and report["not_run"] == 23
    assert report["strict_correct"] == 0 and report["tokens"] is None
    assert report["provider_errors"] == 1


def test_messages_contain_input_without_answers_or_untrusted_system_roles():
    turns = turns_for("我爸喜欢小米，给我推荐手机")
    messages = build_messages(turns, ())
    assert [m["role"] for m in messages] == ["system", "user"]
    payload = json.loads(messages[1]["content"])
    assert payload["turns"][0]["content"] == "我爸喜欢小米，给我推荐手机"
    assert "expected" not in payload
    assert payload["turns"][0]["codepoints"][0] == [0, "我"]


def test_default_cli_no_key_no_network_and_live_missing_key_is_clear(tmp_path):
    env = os.environ.copy()
    env.pop("DEEPSEEK_API_KEY", None)
    dry = subprocess.run(
        [sys.executable, "scripts/evaluate_memory_extraction.py"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert dry.returncode == 0
    assert json.loads(dry.stdout)["api_calls"] == 0
    live = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_memory_extraction.py",
            "--live",
            "--output-dir",
            str(tmp_path / "run"),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert live.returncode != 0 and "DEEPSEEK_API_KEY" in live.stderr
    assert not (tmp_path / "run").exists()


def test_real_run_never_overwrites_existing_artifacts_or_exceeds_24(tmp_path):
    cases = load_cases(FIXTURE)
    with pytest.raises(ValueError):
        evaluate_cases(
            cases + (cases[0],),
            complete=lambda _: pytest.fail("不得调用"),
            output_dir=tmp_path / "too-many",
        )
    old = tmp_path / "existing"
    old.mkdir()
    (old / "keep").write_text("preserve")
    with pytest.raises(FileExistsError):
        evaluate_cases(cases, complete=lambda _: pytest.fail("不得调用"), output_dir=old)
    assert (old / "keep").read_text() == "preserve"
