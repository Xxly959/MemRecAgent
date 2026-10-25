"""完整生命周期参考和固定客户端控制，不冒充真实模型表现。"""

import json
from pathlib import Path

import pytest
from test_memory_analyst_evaluation import paired_client

from memrecagent.memory_interaction_evaluation import evaluate_interactions, load_interactions

FIXTURE = Path(__file__).parent / "fixtures/memory_lifecycle_cases_v1.json"


@pytest.mark.parametrize("mode", ["direct", "analyst"])
def test_lifecycle_fixture_complete_with_fixed_valid_responses(tmp_path, mode):
    cases = load_interactions(FIXTURE)
    seen = []
    report = evaluate_interactions(
        cases, complete=paired_client(cases, seen), output_dir=tmp_path / mode, analysis_mode=mode
    )
    assert report["strict_steps"] == 18 and report["strict_episodes"] == 3
    assert report["confirmation_correct"] == 2
    assert report["model_calls"] == (20 if mode == "direct" else 40)
    assert (
        report["processing_errors"]
        == report["wrong_written_records"]
        == report["missing_updates"]
        == 0
    )
    assert all(
        not any(key in json.dumps(p) for key in ('"expected"', '"heads"', '"allowed_writes"'))
        for p in seen
    )
    matching = [
        p
        for p in seen
        if p["latest_user_turn_id"] == "T3" and p["turns"][-1]["content"] == "给我姨妈挑手机吧。"
    ]
    assert matching and all(
        p["output_contract"]["known_other_subjects"]
        == [{"kind": "other", "references": [{"turn_id": "T2", "quote": "我姨妈"}]}]
        for p in matching
    )
    rows = [json.loads(line) for line in (tmp_path / mode / "steps.jsonl").read_text().splitlines()]
    assert next(
        r for r in rows if r["episode_id"] == "Q_people_override_revision" and r["turn_id"] == "T3"
    )["context_people"] == [["我姨妈", "phone", "苹果", "prefer"]]
