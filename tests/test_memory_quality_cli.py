"""语言评测默认不消费Key；live必须显式预算，语料按完整故事隔离。"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def cli(tmp_path, *args):
    environment = os.environ.copy()
    environment.pop("DEEPSEEK_API_KEY", None)
    return subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_memory_quality.py",
            "--output-dir",
            str(tmp_path / "run"),
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=15,
        env=environment,
    )


def test_default_cli_validates_corpus_without_key_calls_or_output_directory(tmp_path):
    process = cli(tmp_path)
    assert process.returncode == 0, process.stderr
    plan = json.loads(process.stdout)
    assert plan["api_calls"] == 0 and plan["split"] == "dev"
    assert plan["n_episodes"] > 1 and plan["n_steps"] > plan["n_episodes"]
    assert plan["max_provider_attempts"] == plan["n_steps"]
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("args", [["--live"], ["--live", "--max-calls", "1"]])
def test_live_without_explicit_budget_or_key_never_starts(args, tmp_path):
    process = cli(tmp_path, *args)
    assert process.returncode == 2
    assert json.loads(process.stderr)["error"] in {
        "explicit_budget_required",
        "missing_environment_key",
    }
    assert not (tmp_path / "run").exists()


def test_corpus_has_both_splits_diverse_failure_classes_and_no_template_family_overlap():
    from memrecagent.memory_quality_data import load_quality_suite

    data = load_quality_suite(Path("tests/fixtures/memory_quality_v1.json"))
    dev = {e["template_family"] for e in data["episodes"] if e["split"] == "dev"}
    test = {e["template_family"] for e in data["episodes"] if e["split"] == "test"}
    assert dev and test and not dev & test
    tags = {tag for e in data["episodes"] for s in e["steps"] for tag in s["tags"]}
    assert {
        "injection",
        "other_person",
        "restart",
        "revision",
        "neutral",
        "expired",
        "clarification",
        "current_override",
        "sarcasm",
        "trusted_delete",
    } <= tags
    assert all(
        not s["expected"]["writes"]
        for e in data["episodes"]
        for s in e["steps"]
        if "injection" in s["tags"]
    )
