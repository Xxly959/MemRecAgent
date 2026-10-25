"""保护真实未完整表达，防止用空断言数组吞掉必要澄清。"""

import json
import os
import subprocess
import sys
from datetime import timedelta

import pytest
from test_memory_extraction import NOW, output_for, span, turns_for

from memrecagent.memory_commit import CommitError, MemoryCommitSession
from memrecagent.memory_extraction import decode_candidate
from memrecagent.memory_store import JsonlMemoryStore


@pytest.mark.parametrize("field", ["entity", "relation"])
def test_unresolved_only_preference_still_blocks_commit_without_assertion(tmp_path, field):
    text = "我以后喜欢那个品牌，名字还没说。" if field == "entity" else "我对小米的态度还没说完整。"
    turns = turns_for(text)
    value = output_for(turns[0])
    value["assertions"] = []
    value["unresolved"] = [{"field": field, "evidence": [span(turns[0])]}]
    store = JsonlMemoryStore(tmp_path)
    session = MemoryCommitSession(store, user_id="synthetic-user", session_id="S1")
    as_of = NOW + timedelta(hours=1)
    preview = session.prepare(
        decode_candidate(json.dumps(value), turns=turns), turns=turns, as_of=as_of
    )
    assert preview.action == "clarify"
    assert preview.proposal.proposed_memories == ()
    with pytest.raises(CommitError, match="confirmation_required"):
        session.commit(preview.ticket_id, as_of=as_of)
    assert store.list_memories("synthetic-user", as_of=as_of) == ()


def test_no_request_attitude_or_unresolved_is_noop_and_cannot_create_memory(tmp_path):
    turns = turns_for("我爸和我朋友在看手机，都还没说喜欢什么。")
    value = output_for(turns[0])
    value["assertions"] = []
    store = JsonlMemoryStore(tmp_path)
    session = MemoryCommitSession(store, user_id="synthetic-user", session_id="S1")
    as_of = NOW + timedelta(hours=1)
    preview = session.prepare(
        decode_candidate(json.dumps(value), turns=turns), turns=turns, as_of=as_of
    )
    assert preview.action == "noop"
    assert session.commit(preview.ticket_id, as_of=as_of).status == "not_applied"
    assert store.list_memories("synthetic-user", as_of=as_of) == ()


def test_v3_plan_is_fixed_synthetic_suite_without_key_or_output(tmp_path):
    env = os.environ.copy()
    env.pop("DEEPSEEK_API_KEY", None)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_memory_dialogues.py",
            "--suite",
            "acceptance-v3",
            "--output-dir",
            str(tmp_path / "run"),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "mode": "plan",
        "suite": "acceptance-v3",
        "episodes": 3,
        "base_steps": 18,
        "max_model_calls": 20,
        "api_calls": 0,
    }
    assert not (tmp_path / "run").exists()
