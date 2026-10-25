"""新购买意图生命周期的固定控制；不证明自然语言模型语义通过。"""

import json
from pathlib import Path

import pytest
from test_memory_analyst_evaluation import paired_client

from memrecagent.memory_interaction_evaluation import evaluate_interactions, load_interactions

FIXTURE = Path(__file__).parent / "fixtures/memory_request_lifecycle_v1.json"


@pytest.mark.parametrize("mode", ["direct", "analyst"])
def test_request_lifecycle_actual_states_and_no_reference_leak(tmp_path, mode):
    cases = load_interactions(FIXTURE)
    seen = []
    report = evaluate_interactions(
        cases, complete=paired_client(cases, seen), output_dir=tmp_path / mode, analysis_mode=mode
    )
    assert report["strict_steps"] == 18 and report["strict_episodes"] == 3
    assert report["confirmation_correct"] == 2
    assert report["model_calls"] == (20 if mode == "direct" else 40)
    assert report["wrong_written_records"] == report["missing_updates"] == 0
    assert report["heads_correct"] == report["activation_correct"] == 18
    assert all(
        not any(key in json.dumps(p) for key in ('"expected"', '"heads"', '"allowed_writes"'))
        for p in seen
    )
    rows = [json.loads(line) for line in (tmp_path / mode / "steps.jsonl").read_text().splitlines()]
    kept = next(r for r in rows if r["episode_id"].startswith("U_") and r["turn_id"] == "T4")
    assert kept["heads"] == [["phone", "小米", "exclude"]]
    assert kept["written"] == []
    matched = [
        p for p in seen if p["turns"][-1]["content"] == "我室友要换手机了，给他拿几个合适的选项。"
    ]
    assert matched and all(
        p["output_contract"]["known_other_subjects"]
        == [{"kind": "other", "references": [{"turn_id": "T4", "quote": "我室友"}]}]
        for p in matched
    )
