"""无效草案只是独立分析的数据，最终严格边界不能绕过。"""

import json
from dataclasses import replace

import pytest
from test_memory_analyst import reply, task_for
from test_memory_interaction import after, turn, value
from test_memory_interaction_extraction import quote

from memrecagent.deepseek_client import ModelReply, ProviderError
from memrecagent.memory_analyst import AnalystError, build_analyst_messages, inspect_draft
from memrecagent.memory_commit import CommitError
from memrecagent.memory_interaction import ConversationMemorySession
from memrecagent.memory_interaction_extraction import decode_interaction
from memrecagent.memory_store import JsonlMemoryStore
from memrecagent.memory_use_policy import MemoryUsePolicy


def broken(t, kind):
    if kind == "json":
        return "{"
    data = value(t, entity="三星", relation="prefer")
    if kind == "entity":
        data["candidate"]["assertions"][0]["evidence"] = [quote(t, "我喜欢")]
    elif kind == "historical":
        data["candidate"]["assertions"][0]["evidence"] = [{"turn_id": "missing", "quote": "三星"}]
    return json.dumps(data, ensure_ascii=False)


@pytest.mark.parametrize("kind", ["json", "entity", "historical"])
def test_bad_draft_then_final_can_prepare_but_only_final_can_commit(tmp_path, kind):
    store = JsonlMemoryStore(tmp_path)
    host = ConversationMemorySession(store, user_id="u", session_id="s")
    t = turn(1, "我喜欢三星，以后都优先它。")
    raw = broken(t, kind)
    seen = []

    def complete(messages):
        seen.append(json.loads(messages[-1]["content"]))
        assert not store.list_memories("u", as_of=after(t))
        return (
            ModelReply(raw, "fake", None, {}, 0)
            if len(seen) == 1
            else reply(value(t, entity="三星", relation="prefer"))
        )

    with pytest.raises(ValueError):
        decode_interaction(raw, turns=(t,))
    p = host.prepare_user(t, complete=complete, as_of=after(t), analysis_mode="analyst")
    assert len(seen) == 2 and seen[1]["draft"] is None
    assert seen[1]["untrusted_draft"]["raw"] == raw
    assert seen[1]["memory_view"]["entries"] == []
    assert [c.status for c in p.analysis_calls] == ["processing_error", "ok"]
    assert not store.list_memories("u", as_of=after(t))
    host.commit(p.ticket_id, as_of=after(t))
    assert [(m.entity, m.relation) for m in store.active_memories("u", as_of=after(t))] == [
        ("三星", "prefer")
    ]


@pytest.mark.parametrize(
    "bad",
    [
        ProviderError("transport"),
        None,
        ModelReply("{", "fake", None, {"total_tokens": True}, 0),
        ModelReply("x" * 65537, "fake", None, {}, 0),
    ],
)
def test_wire_metadata_and_oversize_do_not_enter_second_stage(tmp_path, bad):
    host = ConversationMemorySession(JsonlMemoryStore(tmp_path), user_id="u", session_id="s")
    t = turn(1, "我喜欢三星。")
    calls = []

    def complete(messages):
        calls.append(messages)
        if isinstance(bad, Exception):
            raise bad
        return bad

    with pytest.raises(AnalystError):
        host.prepare_user(t, complete=complete, as_of=after(t), analysis_mode="analyst")
    assert len(calls) == 1 and len(host.last_analysis_calls) == 1


@pytest.mark.parametrize("final", ["{", None, ProviderError("transport")])
def test_bad_final_never_retries_or_falls_back(tmp_path, final):
    store = JsonlMemoryStore(tmp_path)
    host = ConversationMemorySession(store, user_id="u", session_id="s")
    t = turn(1, "我喜欢三星。")
    calls = []

    def complete(messages):
        calls.append(messages)
        if len(calls) == 1:
            return ModelReply("{", "fake", None, {}, 0)
        if isinstance(final, Exception):
            raise final
        return ModelReply(final, "fake", None, {}, 0) if final is not None else None

    with pytest.raises(AnalystError):
        host.prepare_user(t, complete=complete, as_of=after(t), analysis_mode="analyst")
    assert len(calls) == 2 and len(host.last_analysis_calls) == 2
    assert not store.list_memories("u", as_of=after(t))


def test_inspection_retains_processing_cost_without_repr_leaking_raw():
    t = turn(1, "我喜欢三星。")
    result = inspect_draft((t,), complete=lambda _: ModelReply("私密草案{", "fake", None, {}, 0))
    assert result.analysis is None and result.call.status == "processing_error"
    assert result.raw == "私密草案{" and result.error_code == "interaction_json"
    assert "私密草案" not in repr(result)


@pytest.mark.parametrize("change", ["false_marker", "wrong_code", "old_memory"])
def test_invalid_draft_marker_cannot_smuggle_valid_draft_or_memory(change):
    from test_memory_analyst import memory

    from memrecagent.memory_analyst import MemoryView

    t = turn(1, "我喜欢三星。")
    task = task_for(t)
    if change == "false_marker":
        task = replace(task, draft_error_code="interaction_json")
    else:
        task = replace(
            task,
            draft_raw="{",
            draft_error_code="entity_evidence" if change == "wrong_code" else "interaction_json",
        )
        if change == "old_memory":
            task = replace(
                task,
                memory_view=MemoryView("u", "s", after(t), "snapshot-private", (memory(),), False),
            )
    with pytest.raises(AnalystError):
        build_analyst_messages(task)


@pytest.mark.parametrize("mutation", ["external_policy", "same_policy_aba", "delete"])
def test_bad_draft_with_inflight_invalidation_stops_before_final(tmp_path, mutation):
    store = JsonlMemoryStore(tmp_path)
    host = ConversationMemorySession(store, user_id="u", session_id="s")
    other = ConversationMemorySession(store, user_id="u", session_id="other")
    t = turn(1, "我喜欢三星。")
    calls = []

    def complete(messages):
        calls.append(messages)
        if mutation == "delete":
            store.delete_entity("u", category="phone", entity="三星", deleted_at=after(t))
        elif mutation == "same_policy_aba":
            host.set_use_policy(MemoryUsePolicy(False, False, False))
            host.set_use_policy(MemoryUsePolicy())
        else:
            other.set_use_policy(MemoryUsePolicy(False, False, False))
        return ModelReply("{", "fake", None, {}, 0)

    with pytest.raises(CommitError, match="stale"):
        host.prepare_user(t, complete=complete, as_of=after(t), analysis_mode="analyst")
    assert len(calls) == 1 and len(host.last_analysis_calls) == 1
    assert not host.turns and not host.working_clues


def test_invalid_callback_is_rejected_before_recording_a_call():
    t = turn(1, "我喜欢三星。")
    with pytest.raises(AnalystError) as found:
        inspect_draft((t,), complete=None)
    assert found.value.code == "client" and found.value.call is None


def test_task_repr_does_not_echo_untrusted_error_marker():
    task = replace(task_for(turn(1, "随便看看")), draft_error_code="私密的伪错误码")
    assert "私密的伪错误码" not in repr(task)


def test_valid_wire_but_unbound_confirmation_is_not_a_decode_failure():
    from test_memory_interaction_extraction import confirmation_value

    q = turn(1, "这次还是长期？", "assistant")
    t = turn(2, "我本人的长期偏好。")
    data = confirmation_value((turn(0, "我喜欢三星，范围未定。"), q, t))
    data["confirmation"]["question_turn_id"] = q.turn_id
    data["confirmation"]["evidence"] = [quote(t)]
    calls = []

    def complete(messages):
        calls.append(messages)
        return reply(data)

    with pytest.raises(AnalystError) as found:
        inspect_draft((q, t), complete=complete)
    assert found.value.code == "interaction_question" and len(calls) == 1


def test_bad_draft_cannot_select_existing_private_memory(tmp_path):
    from test_memory import sample_memory

    store = JsonlMemoryStore(tmp_path)
    store.add(sample_memory(user_id="u", entity="三星", evidence_text="不应进入无效草案视图"))
    host = ConversationMemorySession(store, user_id="u", session_id="s")
    t = turn(1, "你怎么看三星？")
    seen = []

    def complete(messages):
        seen.append(json.loads(messages[-1]["content"]))
        return (
            ModelReply("{", "fake", None, {}, 0)
            if len(seen) == 1
            else reply(value(t, entity="三星"))
        )

    p = host.prepare_user(t, complete=complete, as_of=after(t), analysis_mode="analyst")
    assert len(seen) == 2 and seen[1]["memory_view"]["entries"] == []
    assert "不应进入无效草案视图" not in json.dumps(seen, ensure_ascii=False)
    assert p.context_memories[0].entity == "三星"


def test_standalone_strict_draft_function_still_rejects_bad_json():
    from memrecagent.memory_analyst import analyze_draft

    t = turn(1, "我喜欢三星。")
    with pytest.raises(AnalystError):
        analyze_draft((t,), complete=lambda _: ModelReply("{", "fake", None, {}, 0))


def test_evaluator_does_not_hide_rejected_draft_when_final_step_succeeds(tmp_path):
    from pathlib import Path

    from test_memory_analyst_evaluation import paired_client

    from memrecagent.memory_interaction_evaluation import evaluate_interactions, load_interactions

    cases = load_interactions(Path(__file__).parent / "fixtures/memory_analyst_cases_v1.json")
    fixed = paired_client(cases)
    sent = []

    def complete(messages):
        sent.append(messages)
        response = fixed(messages)
        return ModelReply("{", "fake", None, {}, 0) if len(sent) == 1 else response

    report = evaluate_interactions(
        cases, complete=complete, output_dir=tmp_path / "run", analysis_mode="analyst"
    )
    assert report["processing_errors"] == 0 and report["strict_steps"] == 18
    assert report["model_calls"] == 40 and report["draft_processing_errors"] == 1
    records = [
        json.loads(line)
        for line in (tmp_path / "run" / "analysis-records.jsonl").read_text().splitlines()
    ]
    assert len(records) == 40
    assert records[0]["stage"] == "draft" and records[0]["status"] == "processing_error"
    assert records[1]["stage"] == "analyst" and records[1]["status"] == "ok"
