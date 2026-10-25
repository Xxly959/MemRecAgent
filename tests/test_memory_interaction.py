"""真实存储和宿主的连续对话测试；固定模型返回不代表理解准确率。"""

import json
from datetime import timedelta

import pytest
from test_memory_extraction import NOW, subject
from test_memory_interaction_extraction import envelope, quote

from memrecagent.deepseek_client import ModelReply
from memrecagent.memory_commit import CommitError
from memrecagent.memory_extraction import ExtractionError, Turn
from memrecagent.memory_interaction import ConversationMemorySession
from memrecagent.memory_store import JsonlMemoryStore


def turn(i, text, role="user"):
    time = NOW + timedelta(minutes=i)
    return Turn(f"T{i}", role, text, time, time)


def after(t):
    return t.available_at + timedelta(seconds=1)


def value(t, intent="consult", *, entity=None, relation=None, scope="long_term", clue=False):
    raw = envelope()
    raw["intents"] = [intent]
    if entity and entity in t.content:
        raw["topics"] = [quote(t, entity)]
    row = {"subject": subject(), "category": "phone", "entity": entity, "evidence": [quote(t)]}
    if relation:
        raw["intents"] = ["express_preference"]
        raw["candidate"]["assertions"] = [dict(row, relation=relation, scope=scope)]
    if clue:
        raw["clues"] = [row]
    if intent == "recommend":
        raw["candidate"]["request"].update(
            is_recommendation=True,
            beneficiary=subject(),
            category="phone",
            evidence=[quote(t)],
            requested_entity=entity,
        )
    return raw


def model(raw, seen=None):
    def complete(messages):
        if seen is not None:
            seen.append(json.loads(messages[-1]["content"]))
        return ModelReply(json.dumps(raw, ensure_ascii=False), "fake", None, None, 0.0)

    return complete


def prepare(session, t, raw, seen=None):
    return session.prepare_user(t, complete=model(raw, seen), as_of=after(t))


def accept(session, t, raw):
    preview = prepare(session, t, raw)
    session.commit(preview.ticket_id, as_of=after(t))
    return preview


@pytest.fixture
def setup(tmp_path):
    store = JsonlMemoryStore(tmp_path / "memory")
    return store, ConversationMemorySession(store, user_id="u", session_id="s")


def active(store, t):
    return store.active_memories("u", as_of=after(t))


def test_unrelated_consult_then_explicit_preference_and_later_activation(setup):
    store, session = setup
    t = turn(1, "今天下雨了")
    assert accept(session, t, value(t, "unrelated")).response_action == "continue"
    t = turn(2, "你怎么看三星？")
    assert accept(session, t, value(t, entity="三星")).response_action == "answer"
    assert not active(store, t)
    session.observe_assistant(turn(3, "三星系统功能比较丰富。", "assistant"))
    t = turn(4, "三星系统我用着顺手")
    p = prepare(session, t, value(t, entity="三星", clue=True))
    assert p.memory_preview.action != "clarify"
    assert not session.working_clues
    session.commit(p.ticket_id, as_of=after(t))
    assert len(session.working_clues) == 1 and not active(store, t)
    t = turn(5, "我一直更喜欢三星手机")
    p = prepare(session, t, value(t, entity="三星", relation="prefer"))
    assert not active(store, t)
    session.commit(p.ticket_id, as_of=after(t))
    assert [(m.entity, m.relation) for m in active(store, t)] == [("三星", "prefer")]
    assert not session.working_clues
    t = turn(6, "晚上吃什么")
    assert not accept(session, t, value(t, "unrelated")).context_memories
    t = turn(7, "给我推荐手机")
    p = accept(session, t, value(t, "recommend"))
    assert p.response_action == "recommend"
    assert [m.entity for m in p.context_memories] == ["三星"]
    t = turn(8, "这次只看小米手机")
    accept(session, t, value(t, "recommend", entity="小米"))
    assert [(m.entity, m.relation) for m in active(store, t)] == [("三星", "prefer")]
    t = turn(9, "我现在取消对三星的偏好")
    accept(session, t, value(t, entity="三星", relation="neutral"))
    assert [m.relation for m in active(store, t)] == ["neutral"]
    assert len(store.list_memories("u", as_of=after(t))) == 2


def test_assistant_praise_and_repeated_questions_never_create_self_preference(setup):
    store, session = setup
    for i in (1, 3, 5):
        t = turn(i, "三星好吗？")
        accept(session, t, value(t, entity="三星"))
        session.observe_assistant(turn(i + 1, "三星很棒", "assistant"))
    t = turn(7, "我买过三星手机")
    accept(session, t, value(t, entity="三星", clue=True))
    assert not active(store, t)
    assert len(session.working_clues) == 1


def pending(session):
    t = turn(1, "我喜欢三星")
    p = prepare(session, t, value(t, entity="三星", relation="prefer", scope="unclear"))
    assert p.response_action == "clarify"
    q = turn(2, p.memory_preview.proposal.decision.clarification, "assistant")
    return p, q


def confirmation(t, q, outcome="confirm"):
    raw = value(t)
    raw["confirmation"] = dict(
        question_turn_id=q.turn_id,
        outcome=outcome,
        subject="self" if outcome == "confirm" else None,
        scope="long_term" if outcome == "confirm" else None,
        person_reference=None,
        evidence=[quote(t)],
    )
    return raw


def test_single_actual_question_can_bind_short_confirmation_with_role_provenance(setup):
    store, session = setup
    _, q = pending(session)
    session.observe_assistant(q)
    t = turn(3, "后者")
    p = prepare(session, t, confirmation(t, q))
    assert not active(store, t)
    session.commit(p.ticket_id, as_of=after(t))
    (memory,) = active(store, t)
    assert "我喜欢三星" in memory.evidence_text and "后者" in memory.evidence_text
    assert "assistant语境" in memory.evidence_text and q.content in memory.evidence_text
    assert q.turn_id in memory.evidence_ref


@pytest.mark.parametrize("kind", ["ordinary", "old", "generic", "wrong_order"])
def test_unbound_or_ambiguous_confirmation_never_writes(setup, kind):
    store, session = setup
    _, q = pending(session)
    if kind == "ordinary":
        q = turn(2, "你要三星还是苹果？还有预算多少？", "assistant")
    session.observe_assistant(q)
    if kind == "old":
        session.observe_assistant(turn(3, "我们换个话题", "assistant"))
    t = turn(4, "嗯" if kind == "generic" else "前者" if kind == "wrong_order" else "后者")
    with pytest.raises(CommitError):
        prepare(session, t, confirmation(t, q))
    assert not active(store, t)


def test_new_topic_does_not_confirm_pending_change(setup):
    store, session = setup
    old, q = pending(session)
    session.observe_assistant(q)
    t = turn(3, "今天有雨吗")
    p = accept(session, t, value(t, "unrelated"))
    assert p.response_action == "continue" and not active(store, t)
    with pytest.raises(CommitError, match="ticket"):
        session.commit(old.ticket_id, as_of=after(t))


def test_consultation_does_not_turn_old_exclusion_into_refusal(setup):
    store, session = setup
    t = turn(1, "我不想再买小米手机")
    accept(session, t, value(t, entity="小米", relation="exclude"))
    t = turn(2, "你怎么看小米")
    p = accept(session, t, value(t, entity="小米"))
    assert p.response_action == "answer" and p.memory_preview.action != "clarify"
    assert [(m.entity, m.relation) for m in p.context_memories] == [("小米", "exclude")]
    assert len(active(store, t)) == 1


def test_item_clue_does_not_become_brand_dislike(setup):
    store, session = setup
    t = turn(1, "小米这台手机发热")
    p = accept(session, t, value(t, entity="小米", clue=True))
    assert p.response_action == "answer" and not active(store, t)
    assert len(session.working_clues) == 1
    other = ConversationMemorySession(store, user_id="u", session_id="s")
    assert not other.working_clues and not other.turns


def test_cancel_and_repeated_commit_do_not_publish_partial_clues(setup):
    _, session = setup
    t = turn(1, "三星系统不错")
    p = prepare(session, t, value(t, entity="三星", clue=True))
    session.cancel(p.ticket_id)
    assert not session.working_clues
    with pytest.raises(CommitError):
        session.commit(p.ticket_id, as_of=after(t))
    t = turn(2, "小米系统不错")
    p = accept(session, t, value(t, entity="小米", clue=True))
    session.commit(p.ticket_id, as_of=after(t))
    assert len(session.working_clues) == 1


def test_storage_failure_does_not_publish_clues(setup, monkeypatch):
    store, session = setup
    t = turn(1, "我喜欢三星，小米这台有点热")
    raw = value(t, entity="三星", relation="prefer")
    raw["clues"] = value(t, entity="小米", clue=True)["clues"]
    p = prepare(session, t, raw)
    original = store.add_many

    def fail(*args, **kwargs):
        raise OSError("模拟磁盘失败")

    monkeypatch.setattr(store, "add_many", fail)
    with pytest.raises(CommitError, match="storage"):
        session.commit(p.ticket_id, as_of=after(t))
    assert not session.working_clues and not active(store, t)
    monkeypatch.setattr(store, "add_many", original)
    session.commit(p.ticket_id, as_of=after(t))
    assert len(session.working_clues) == 1 and len(active(store, t)) == 1


@pytest.mark.parametrize("bad", ["provider", "json", "reply_type"])
def test_model_failure_keeps_actual_prefix_but_invalidates_ticket(setup, bad):
    store, session = setup
    t = turn(1, "三星系统不错")
    p = prepare(session, t, value(t, entity="三星", clue=True))
    calls = []

    def complete(messages):
        calls.append(messages)
        if bad == "provider":
            raise TimeoutError("模拟超时")
        return "bad" if bad == "reply_type" else ModelReply("{", "fake", None, None, 0.0)

    t = turn(2, "三星好吗")
    with pytest.raises((TimeoutError, ExtractionError, CommitError)):
        session.prepare_user(t, complete=complete, as_of=after(t))
    assert len(calls) == 1 and session.turns[-1] == t
    assert not active(store, t) and not session.working_clues
    with pytest.raises(CommitError):
        session.commit(p.ticket_id, as_of=after(t))


def test_actual_assistant_history_and_no_unrelated_memory_in_classifier(setup):
    _, session = setup
    t = turn(1, "我喜欢三星")
    accept(session, t, value(t, entity="三星", relation="prefer"))
    a = turn(2, "它的系统你用过吗？", "assistant")
    session.observe_assistant(a)
    t = turn(3, "今天下雨")
    seen = []
    p = prepare(session, t, value(t, "unrelated"), seen)
    assert seen[0]["own_memories"] == []
    assert seen[0]["turns"][-2] == {"turn_id": a.turn_id, "role": "assistant", "content": a.content}
    assert not p.context_memories


@pytest.mark.parametrize("kind", ["future", "role", "duplicate"])
def test_invalid_turn_rejected_before_model(setup, kind):
    _, session = setup
    t = turn(1, "你好")
    accept(session, t, value(t, "unrelated"))
    t = t if kind == "duplicate" else turn(2, "你好", "assistant" if kind == "role" else "user")
    calls = []
    with pytest.raises((CommitError, ExtractionError)):
        session.prepare_user(
            t,
            complete=model(value(t), calls),
            as_of=t.available_at if kind == "future" else after(t),
        )
    assert not calls


def test_feedback_requires_trusted_target(setup):
    store, session = setup
    t = turn(1, "不感兴趣")
    p = accept(session, t, value(t, "feedback"))
    assert p.response_action == "feedback_target_required" and not active(store, t)


def test_delete_resets_working_prefix_before_any_model_call(setup):
    store, session = setup
    t = turn(1, "我喜欢三星")
    accept(session, t, value(t, entity="三星", relation="prefer"))
    t = turn(2, "小米这台有点热")
    accept(session, t, value(t, entity="小米", clue=True))
    store.delete_entity("u", category="phone", entity="三星", deleted_at=after(turn(3, "删除")))
    t = turn(4, "今天下雨")
    seen = []
    with pytest.raises(CommitError, match="stale"):
        prepare(session, t, value(t, "unrelated"), seen)
    assert not seen and not session.turns and not session.working_clues
    prepare(session, t, value(t, "unrelated"), seen)
    assert [r["turn_id"] for r in seen[0]["turns"]] == [t.turn_id]
    assert not active(store, t)


def test_external_change_during_model_call_is_stale_and_drops_returned_candidate(setup):
    store, session = setup
    t = turn(1, "我喜欢三星")

    def complete(messages):
        store.delete_entity("u", category="phone", entity="三星", deleted_at=after(t))
        return model(value(t, entity="三星", relation="prefer"))(messages)

    with pytest.raises(CommitError, match="stale"):
        session.prepare_user(t, complete=complete, as_of=after(t))
    assert not session.turns and not active(store, t)


def test_stale_commit_does_not_publish_clues(setup):
    store, session = setup
    t = turn(1, "三星系统不错")
    p = prepare(session, t, value(t, entity="三星", clue=True))
    store.delete_entity("u", category="phone", entity="三星", deleted_at=after(t))
    with pytest.raises(CommitError, match="stale"):
        session.commit(p.ticket_id, as_of=after(t))
    assert not session.working_clues


def test_storage_root_rebinding_never_switches_owner(setup, tmp_path):
    store, session = setup
    original = store.root
    t = turn(1, "三星系统不错")
    accept(session, t, value(t, entity="三星", clue=True))
    store.root = tmp_path / "other"
    t = turn(2, "你好")
    with pytest.raises(CommitError, match="binding"):
        prepare(session, t, value(t, "unrelated"))
    assert not session.turns and not session.working_clues
    store.root = original
    assert prepare(session, t, value(t, "unrelated")).response_action == "continue"


def test_explicit_other_beneficiary_does_not_receive_own_background(setup):
    store, session = setup
    t = turn(1, "我喜欢三星")
    accept(session, t, value(t, entity="三星", relation="prefer"))
    t = turn(2, "帮我爸看一下三星")
    raw = value(t, entity="三星")
    raw["candidate"]["request"]["beneficiary"] = subject("other", [quote(t, "我爸")])
    p = accept(session, t, raw)
    assert not p.context_memories and len(active(store, t)) == 1


def test_others_stable_preference_only_lives_in_accepted_session(setup):
    store, session = setup
    t = turn(1, "我爸一直喜欢小米")
    raw = value(t, entity="小米", relation="prefer")
    who = subject("other", [quote(t, "我爸")])
    raw["candidate"]["assertions"][0]["subject"] = who
    accept(session, t, raw)
    assert not active(store, t)
    t = turn(2, "给我爸推荐手机")
    raw = value(t, "recommend")
    raw["candidate"]["request"]["beneficiary"] = who
    p = accept(session, t, raw)
    assert not p.context_memories
    assert [c.entity for c in p.memory_preview.proposal.decision.context_preferences] == ["小米"]
    fresh = ConversationMemorySession(store, user_id="u", session_id="different")
    t = turn(3, "给我爸推荐手机")
    raw = value(t, "recommend")
    raw["candidate"]["request"]["beneficiary"] = subject("other", [quote(t, "我爸")])
    p = prepare(fresh, t, raw)
    assert not p.memory_preview.proposal.decision.context_preferences


def test_unasked_or_future_question_does_not_enable_confirmation(setup):
    store, session = setup
    _, q = pending(session)
    t = turn(3, "后者")
    with pytest.raises(ExtractionError, match="interaction_question"):
        prepare(session, t, confirmation(t, q))
    assert not active(store, t)


def test_clue_removed_after_confirmed_self_preference(setup):
    _, session = setup
    t = turn(0, "三星系统顺手")
    accept(session, t, value(t, entity="三星", clue=True))
    _, q = pending(session)
    session.observe_assistant(q)
    t = turn(3, "后者")
    accept(session, t, confirmation(t, q))
    assert not session.working_clues


def test_confirmation_cancel_and_unclear_do_not_save(setup):
    store, session = setup
    _, q = pending(session)
    session.observe_assistant(q)
    t = turn(3, "还不确定")
    p = prepare(session, t, confirmation(t, q, "unclear"))
    assert p.response_action == "clarify"
    with pytest.raises(CommitError, match="confirmation_required"):
        session.commit(p.ticket_id, as_of=after(t))
    q = turn(4, p.memory_preview.proposal.decision.clarification, "assistant")
    session.observe_assistant(q)
    t = turn(5, "取消")
    accept(session, t, confirmation(t, q, "cancel"))
    assert not active(store, t)


@pytest.mark.parametrize("invalid", ["role", "content", "duplicate", "future", "late_available"])
def test_core_optional_question_validates_provenance_before_writing(setup, invalid):
    from dataclasses import replace

    from memrecagent.memory_commit import MemoryCommitSession
    from memrecagent.memory_interaction_extraction import decode_interaction

    store, _ = setup
    host = MemoryCommitSession(store, user_id="u", session_id="s")
    original = turn(1, "我喜欢三星")
    candidate = decode_interaction(
        json.dumps(value(original, entity="三星", relation="prefer", scope="unclear")),
        turns=(original,),
    ).candidate
    p = host.prepare(candidate, turns=(original,), as_of=after(original))
    q = turn(2, p.proposal.decision.clarification, "assistant")
    reply = turn(3, "后者")
    if invalid == "role":
        q = replace(q, role="user")
    elif invalid == "content":
        q = replace(q, content="还有别的问题吗？")
    elif invalid == "duplicate":
        q = replace(q, turn_id=original.turn_id)
    elif invalid == "future":
        q = turn(4, q.content, "assistant")
    else:
        q = replace(q, available_at=reply.available_at)
    with pytest.raises(CommitError, match="question"):
        host.confirm(
            p.ticket_id,
            reply=reply,
            subject="self",
            scope="long_term",
            question=q,
            as_of=after(reply),
        )
    assert not active(store, reply)


def test_multi_target_question_cannot_apply_all_with_one_confirmation(setup):
    store, session = setup
    t = turn(1, "我喜欢三星，也喜欢苹果")
    raw = value(t, entity="三星", relation="prefer", scope="unclear")
    raw["candidate"]["assertions"] += value(t, entity="苹果", relation="prefer", scope="unclear")[
        "candidate"
    ]["assertions"]
    p = prepare(session, t, raw)
    q = turn(2, p.memory_preview.proposal.decision.clarification, "assistant")
    session.observe_assistant(q)
    t = turn(3, "后者")
    with pytest.raises(CommitError, match="clarification_target"):
        prepare(session, t, confirmation(t, q))
    assert not active(store, t)


def test_deleted_mixed_old_source_cannot_reappear_in_fresh_prefix(setup):
    store, session = setup
    original = turn(1, "我喜欢三星也喜欢苹果")
    raw = value(original, entity="三星", relation="prefer")
    accept(session, original, raw)
    store.delete_entity("u", category="phone", entity="三星", deleted_at=after(turn(2, "删除")))
    t = turn(3, "我仍然喜欢三星")
    with pytest.raises(CommitError, match="stale"):
        prepare(session, t, value(t, entity="三星", relation="prefer"))
    raw = value(t, entity="三星", relation="prefer")
    raw["candidate"]["assertions"][0]["evidence"].append(quote(original))
    with pytest.raises(ExtractionError, match="quote_missing"):
        prepare(session, t, raw)
    assert not active(store, t)


def test_new_user_does_not_activate_another_users_memory(setup):
    store, session = setup
    t = turn(1, "我喜欢三星")
    accept(session, t, value(t, entity="三星", relation="prefer"))
    fresh = ConversationMemorySession(store, user_id="another", session_id="s")
    t = turn(2, "给我推荐手机")
    p = prepare(fresh, t, value(t, "recommend"))
    assert not p.context_memories and not fresh.working_clues


def test_compare_only_activates_relevant_background(setup):
    _, session = setup
    t = turn(1, "我喜欢三星")
    accept(session, t, value(t, entity="三星", relation="prefer"))
    t = turn(2, "说说小米")
    assert not accept(session, t, value(t, entity="小米")).context_memories
    t = turn(3, "三星和苹果有什么区别")
    raw = value(t, "compare", entity="三星")
    raw["topics"].append(quote(t, "苹果"))
    p = accept(session, t, raw)
    assert p.response_action == "compare"
    assert [m.entity for m in p.context_memories] == ["三星"]


def test_deleted_source_retried_as_new_prefix_cannot_restore_clue(setup):
    store, session = setup
    original = turn(1, "三星系统顺手")
    raw = value(original, entity="三星", clue=True)
    p = prepare(session, original, raw)
    store.delete_entity("u", category="phone", entity="三星", deleted_at=after(turn(2, "删除")))
    with pytest.raises(CommitError, match="stale"):
        session.commit(p.ticket_id, as_of=after(turn(3, "接受")))
    assert not session.working_clues and not session.turns
    with pytest.raises(CommitError, match="deleted_evidence"):
        session.prepare_user(original, complete=model(raw), as_of=after(turn(4, "重试")))
    assert not session.working_clues and not session.turns
    current = turn(5, "三星系统我现在用着顺手")
    accept(session, current, value(current, entity="三星", clue=True))
    assert len(session.working_clues) == 1


def test_mixed_old_and_new_clue_sources_do_not_hide_deleted_original(setup):
    store, session = setup
    old = turn(1, "三星系统顺手")
    store.delete_entity("u", category="phone", entity="三星", deleted_at=after(turn(2, "删除")))
    with pytest.raises(CommitError, match="stale"):
        prepare(session, old, value(old, entity="三星", clue=True))
    # 旧前缀未生成线索时仍可作为实际语境输入；后续提取不能用新时间掩盖旧来源。
    session.prepare_user(old, complete=model(value(old)), as_of=after(turn(3, "分析")))
    new = turn(4, "三星现在用着还行")
    raw = value(new, entity="三星", clue=True)
    raw["clues"][0]["evidence"].append(quote(old))
    with pytest.raises(CommitError, match="deleted_evidence"):
        prepare(session, new, raw)
    assert not session.turns and not session.working_clues


def test_own_deletion_does_not_remove_independent_other_clue(setup):
    store, session = setup
    old = turn(1, "我爸觉得三星系统顺手")
    store.delete_entity("u", category="phone", entity="三星", deleted_at=after(turn(2, "删除")))
    with pytest.raises(CommitError, match="stale"):
        prepare(session, old, value(old, entity="三星", clue=True))
    raw = value(old, entity="三星", clue=True)
    raw["clues"][0]["subject"] = subject("other", [quote(old, "我爸")])
    p = session.prepare_user(old, complete=model(raw), as_of=after(turn(3, "分析")))
    session.commit(p.ticket_id, as_of=after(turn(3, "接受")))
    assert len(session.working_clues) == 1
    assert not active(store, turn(4, "读取"))
