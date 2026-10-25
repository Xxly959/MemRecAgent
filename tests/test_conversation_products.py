"""实际双阶段记忆宿主到完整候选排名，只替换供应商响应。"""

from datetime import timedelta

import pytest
from test_memory_analyst_host import FixedStages
from test_memory_extraction import subject
from test_memory_interaction import after, confirmation, turn, value
from test_memory_interaction_extraction import quote
from test_memory_products import engine

from memrecagent.conversation_products import ConversationProductSession
from memrecagent.memory_commit import CommitError
from memrecagent.memory_store import JsonlMemoryStore


def session(tmp_path, session_id="s"):
    ranker = engine(tmp_path / "memory")
    return ConversationProductSession(
        ranker._service,
        JsonlMemoryStore(tmp_path / "memory"),
        ranker._facts,
        facts_bundle_id=ranker._service.bundle_id,
        user_id="u",
        session_id=session_id,
    )


def process(host, t, raw):
    return host.process_user(
        t, complete=FixedStages(raw, raw), as_of=after(t), strategy="popularity", k=3
    )


def test_actual_analyst_commits_self_and_fresh_session_ranks_from_disk(tmp_path):
    host = session(tmp_path)
    t = turn(1, "我长期排除品牌A。我长期更喜欢品牌C。")
    raw = value(t, entity="品牌A", relation="exclude")
    raw["candidate"]["assertions"] += value(t, entity="品牌C", relation="prefer")["candidate"][
        "assertions"
    ]
    accepted = process(host, t, raw)
    assert accepted["response_action"] == "answer" and accepted["ranking"] is None
    assert len(accepted["new_memory_ids"]) == 2 and len(accepted["analysis_calls"]) == 2
    reopened = session(tmp_path, "new-session")
    t2 = turn(3, "给我推荐手机")
    result = process(reopened, t2, value(t2, "recommend"))
    assert [p["item_id"] for p in result["ranking"]["items"]] == ["C", "B"]
    assert result["ranking"]["items"][0]["signals"][0]["scope"] == "long_term"
    assert result["new_memory_ids"] == []


def test_own_override_does_not_rewrite_durable_exclusion(tmp_path):
    host = session(tmp_path)
    t = turn(1, "我长期排除品牌A")
    process(host, t, value(t, entity="品牌A", relation="exclude"))
    t2 = turn(3, "这次给我看品牌A，旧偏好别改")
    result = process(host, t2, value(t2, "recommend", entity="品牌A"))
    assert [p["item_id"] for p in result["ranking"]["items"]] == ["A"]
    t3 = turn(5, "给我推荐手机")
    again = process(host, t3, value(t3, "recommend"))
    assert [p["item_id"] for p in again["ranking"]["items"]] == ["B", "C"]


def test_other_preference_is_used_now_but_never_written_or_inherited(tmp_path):
    host = session(tmp_path)
    t = turn(1, "我叔叔一直喜欢品牌C，给他推荐手机")
    who = subject("other", [quote(t, "我叔叔")])
    raw = value(t, "recommend")
    raw["candidate"]["request"]["beneficiary"] = who
    raw["intents"].append("express_preference")
    raw["candidate"]["assertions"] = [
        {
            "subject": who,
            "category": "phone",
            "entity": "品牌C",
            "relation": "prefer",
            "scope": "long_term",
            "evidence": [quote(t)],
        }
    ]
    result = process(host, t, raw)
    assert result["new_memory_ids"] == []
    assert [p["item_id"] for p in result["ranking"]["items"]] == ["A", "C", "B"]
    assert result["ranking"]["items"][1]["signals"][0]["scope"] == "current"
    assert JsonlMemoryStore(tmp_path / "memory").list_memories("u", as_of=after(t)) == ()
    fresh = session(tmp_path, "fresh")
    t2 = turn(3, "给我推荐手机")
    assert [
        p["item_id"] for p in process(fresh, t2, value(t2, "recommend"))["ranking"]["items"]
    ] == ["A", "B", "C"]


def test_scope_question_requires_actual_assistant_then_confirmation_before_saving(tmp_path):
    host = session(tmp_path)
    t = turn(1, "我喜欢品牌C，但还没确定要不要长期记住")
    pending = process(host, t, value(t, entity="品牌C", relation="prefer", scope="unclear"))
    assert pending["response_action"] == "clarify" and pending["ranking"] is None
    assert not pending["new_memory_ids"]
    question = turn(2, pending["clarification"], "assistant")
    host.observe_assistant(question)
    reply = turn(3, "这是我本人的长期偏好，请记住")
    result = process(host, reply, confirmation(reply, question))
    assert len(result["new_memory_ids"]) == 1


def test_consult_and_assistant_praise_never_become_preference(tmp_path):
    host = session(tmp_path)
    t = turn(1, "你觉得品牌C怎么样")
    result = process(host, t, value(t, "consult", entity="品牌C"))
    assert result["new_memory_ids"] == [] and result["ranking"] is None
    host.observe_assistant(turn(2, "品牌C可能适合一些需求，仍要看你的实际体验。", "assistant"))
    t2 = turn(3, "给我推荐手机")
    result = process(host, t2, value(t2, "recommend"))
    assert [p["item_id"] for p in result["ranking"]["items"]] == ["A", "B", "C"]


def test_snapshot_and_policy_changes_are_detected_before_further_analysis(tmp_path):
    host = session(tmp_path)
    t = turn(1, "我一直喜欢品牌C")
    process(host, t, value(t, entity="品牌C", relation="prefer"))
    JsonlMemoryStore(tmp_path / "memory").delete_entity(
        "u", category="phone", entity="品牌C", deleted_at=after(t) + timedelta(seconds=1)
    )
    t2 = turn(3, "给我推荐手机")
    with pytest.raises(CommitError, match="stale"):
        host.process_user(
            t2,
            complete=lambda _: pytest.fail("过期不得调用"),
            as_of=after(t2),
            strategy="popularity",
        )


def test_consult_receives_only_relevant_memory_summary_without_original_text(tmp_path):
    host = session(tmp_path)
    t = turn(1, "我长期更喜欢品牌C")
    process(host, t, value(t, entity="品牌C", relation="prefer"))
    t2 = turn(3, "你怎么看品牌C")
    result = process(host, t2, value(t2, "consult", entity="品牌C"))
    assert result["context_memories"][0]["entity"] == "品牌C"
    assert result["context_memories"][0]["relation"] == "prefer"
    assert "evidence_text" not in result["context_memories"][0]


def test_main_model_context_contains_actual_prior_user_and_assistant(tmp_path):
    host = session(tmp_path)
    first = turn(1, "解释召回和排序的差别")
    process(host, first, value(first, "unrelated"))
    answer = turn(2, "第一点是找到候选。第二点是比较候选。", "assistant")
    host.observe_assistant(answer)
    latest = turn(3, "展开刚才第二点")
    result = process(host, latest, value(latest, "unrelated"))
    assert result["conversation_context"] == [
        {"turn_id": first.turn_id, "role": "user", "content": first.content},
        {"turn_id": answer.turn_id, "role": "assistant", "content": answer.content},
    ]
    assert result["history_truncated"] is False


def test_history_view_is_bounded_and_marks_omission_without_faking_text(tmp_path):
    import json

    host = session(tmp_path)
    for i in (1, 3):
        t = turn(i, f"背景{i}")
        process(host, t, value(t, "unrelated"))
        host.observe_assistant(turn(i + 1, "实际已输出段落" * 500, "assistant"))
    t = turn(5, "展开最近说的内容")
    result = process(host, t, value(t, "unrelated"))
    assert result["history_truncated"] is True
    assert len(json.dumps(result["conversation_context"], ensure_ascii=False).encode()) <= 12000
    assert result["conversation_context"][-1]["content"] == "实际已输出段落" * 500


def test_full_analysis_prefix_limit_stops_before_call_instead_of_silent_rewrite(tmp_path):
    host = session(tmp_path)
    for i in (1, 3, 5):
        t = turn(i, "背景")
        process(host, t, value(t, "unrelated"))
        host.observe_assistant(turn(i + 1, "x" * 8000, "assistant"))
    t = turn(7, "继续")
    with pytest.raises(CommitError, match="session_limit"):
        host.process_user(
            t, complete=lambda _: pytest.fail("不应继续调用"), as_of=after(t), strategy="popularity"
        )
    assert host.last_analysis_calls == [] and host.last_new_memory_ids == []
