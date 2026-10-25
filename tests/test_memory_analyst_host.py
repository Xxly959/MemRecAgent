"""双阶段宿主用真实存储验证，只替换外部模型响应。"""

import json
from datetime import timedelta

import pytest
from test_memory import sample_memory
from test_memory_analyst import reply
from test_memory_extraction import subject
from test_memory_interaction import after, confirmation, model, turn, value
from test_memory_interaction_extraction import quote
from test_memory_subject_gate import ambiguous, mixed

from memrecagent.deepseek_client import ModelReply, ProviderError
from memrecagent.memory_analyst import AnalystError
from memrecagent.memory_commit import CommitError
from memrecagent.memory_interaction import ConversationMemorySession
from memrecagent.memory_store import JsonlMemoryStore


def setup(tmp_path, memories=(), **options):
    store = JsonlMemoryStore(tmp_path / "memory")
    for m in memories:
        store.add(m)
    session = ConversationMemorySession(store, user_id="u", session_id="s", **options)
    return store, session


def old(entity="小米", **changes):
    return sample_memory(**(dict(user_id="u", entity=entity) | changes))


class FixedStages:
    """逐次返回外部完整响应，记录真正收到的独立输入。"""

    def __init__(self, *responses, on_call=None):
        self.responses = responses
        self.seen = []
        self.on_call = on_call

    def __call__(self, messages):
        self.seen.append(json.loads(messages[-1]["content"]))
        index = len(self.seen)
        if self.on_call is not None:
            self.on_call(index)
        response = self.responses[index - 1]
        if isinstance(response, Exception):
            raise response
        return reply(response) if isinstance(response, dict) else response


def prepare(session, t, client):
    return session.prepare_user(t, complete=client, as_of=after(t), analysis_mode="analyst")


def test_direct_default_remains_one_call(tmp_path):
    _, session = setup(tmp_path)
    t = turn(1, "今天下雨")
    client = FixedStages(value(t, "unrelated"))
    p = session.prepare_user(t, complete=client, as_of=after(t))
    assert p.response_action == "continue" and len(client.seen) == 1
    assert session.last_analysis_calls == () and p.analysis_calls == ()


@pytest.mark.parametrize("mode", ["", "autonomous", None, True])
def test_invalid_mode_fails_before_input_or_model(tmp_path, mode):
    _, session = setup(tmp_path)
    t = turn(1, "背景")
    with pytest.raises(CommitError, match="analysis_mode"):
        session.prepare_user(
            t, complete=lambda _: pytest.fail("不得调用"), as_of=after(t), analysis_mode=mode
        )
    assert session.turns == ()


def test_unrelated_draft_still_runs_role_and_only_final_candidate_can_commit(tmp_path):
    store, session = setup(tmp_path)
    t = turn(1, "我一直喜欢三星")
    checks = []

    def check(_):
        checks.append(store.list_memories("u", as_of=after(t)))
        assert session.working_clues == ()

    client = FixedStages(
        value(t, "unrelated"), value(t, entity="三星", relation="prefer"), on_call=check
    )
    p = prepare(session, t, client)
    assert checks == [(), ()] and len(client.seen) == 2
    assert client.seen[0]["own_memories"] == []
    assert "memory_view" in client.seen[1] and "own_memories" not in client.seen[1]
    assert client.seen[1]["draft"]["intents"] == ["unrelated"]
    assert not store.list_memories("u", as_of=after(t))
    assert [c.stage for c in p.analysis_calls] == ["draft", "analyst"]
    assert [c.status for c in p.analysis_calls] == ["ok", "ok"]
    assert p.analysis_calls == session.last_analysis_calls
    session.commit(p.ticket_id, as_of=after(t))
    assert [(m.entity, m.relation) for m in store.active_memories("u", as_of=after(t))] == [
        ("三星", "prefer")
    ]


def test_final_noop_discards_strong_draft_and_clues_without_publishing(tmp_path):
    store, session = setup(tmp_path)
    t = turn(1, "我一直喜欢三星")
    draft = value(t, entity="三星", relation="prefer", clue=True)
    p = prepare(session, t, FixedStages(draft, value(t, "unrelated")))
    session.commit(p.ticket_id, as_of=after(t))
    assert p.response_action == "continue" and not p.clues and not session.working_clues
    assert not store.list_memories("u", as_of=after(t))


def test_related_own_memory_and_current_exception_do_not_change_old_attitude(tmp_path):
    prior = old()
    store, session = setup(tmp_path, (prior, old("苹果", memory_id="m2")))
    t = turn(1, "这次给我推荐小米，旧排除不要改")
    data = value(t, "recommend", entity="小米")
    client = FixedStages(data, data)
    p = prepare(session, t, client)
    entries = client.seen[1]["memory_view"]["entries"]
    assert [(m["entity"], m["relation"]) for m in entries] == [("小米", "exclude")]
    assert p.response_action == "recommend"
    assert [m.entity for m in p.context_memories] == ["苹果"]
    session.commit(p.ticket_id, as_of=after(t))
    assert store.active_memories("u", as_of=after(t))[0] == prior


def test_other_recipient_never_reads_or_overwrites_own_preference(tmp_path):
    prior = old()
    store, session = setup(tmp_path, (prior,))
    t = turn(1, "我妈一直喜欢小米，给她推荐手机")
    data = value(t, "recommend")
    data["topics"] = [quote(t, "小米")]
    data["candidate"]["request"]["beneficiary"] = subject("other", [quote(t, "我妈")])
    data["intents"].append("express_preference")
    data["candidate"]["assertions"] = [
        {
            "subject": subject("other", [quote(t, "我妈")]),
            "category": "phone",
            "entity": "小米",
            "relation": "prefer",
            "scope": "long_term",
            "evidence": [quote(t, "我妈一直喜欢小米")],
        }
    ]
    client = FixedStages(data, data)
    p = prepare(session, t, client)
    assert client.seen[1]["memory_view"]["entries"] == []
    session.commit(p.ticket_id, as_of=after(t))
    assert store.list_memories("u", as_of=after(t)) == (prior,)
    assert p.memory_preview.proposal.session_preferences[0].relation == "prefer"


def test_mixed_known_self_fact_and_unknown_recipient_never_partially_publish(tmp_path):
    store, session = setup(tmp_path)
    t = turn(1, "我一直喜欢OPPO，给她推荐手机，但没说她是谁")
    p = prepare(session, t, FixedStages(mixed(t), mixed(t)))
    assert p.response_action == "clarify"
    with pytest.raises(CommitError, match="confirmation_required"):
        session.commit(p.ticket_id, as_of=after(t))
    assert not store.list_memories("u", as_of=after(t))


def test_age_filter_and_user_binding_apply_before_role_view(tmp_path):
    records = (old(), old("三星", user_id="other", memory_id="m2"))
    _, session = setup(tmp_path, records, max_memory_age=timedelta(minutes=30))
    t = turn(1, "给我推荐小米")
    data = value(t, "recommend", entity="小米")
    client = FixedStages(data, data)
    prepare(session, t, client)
    assert client.seen[1]["memory_view"]["entries"] == []
    assert "user_id" not in json.dumps(client.seen[1])


@pytest.mark.parametrize("bad", [ProviderError("transport"), None, reply({}, content="{")])
@pytest.mark.parametrize("stage", [1, 2])
def test_stage_failure_records_attempts_without_draft_fallback(tmp_path, bad, stage):
    store, session = setup(tmp_path)
    first = turn(1, "今天下雨")
    old_ticket = session.prepare_user(
        first, complete=model(value(first, "unrelated")), as_of=after(first)
    )
    t = turn(2, "我一直喜欢三星")
    strong = value(t, entity="三星", relation="prefer")
    recoverable_draft = stage == 1 and isinstance(bad, ModelReply)
    expected_calls = 2 if recoverable_draft else stage
    client = (
        FixedStages(bad, bad)
        if recoverable_draft
        else (FixedStages(bad) if stage == 1 else FixedStages(strong, bad))
    )
    with pytest.raises(AnalystError):
        prepare(session, t, client)
    assert len(client.seen) == expected_calls and session.turns[-1] == t
    assert len(session.last_analysis_calls) == expected_calls
    assert session.last_analysis_calls[-1].status != "ok"
    assert session.last_analysis_calls[-1].stage == ("draft" if expected_calls == 1 else "analyst")
    assert not store.list_memories("u", as_of=after(t)) and not session.working_clues
    with pytest.raises(CommitError, match="ticket"):
        session.commit(old_ticket.ticket_id, as_of=after(t))


@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("mutation", ["delete", "correct"])
def test_external_change_during_either_call_prevents_publication(tmp_path, stage, mutation):
    prior = old()
    store, session = setup(tmp_path, (prior,))
    t = turn(1, "我一直喜欢小米")
    data = value(t, entity="小米", relation="prefer")

    def change(index):
        if index != stage:
            return
        if mutation == "delete":
            store.delete_entity("u", category="phone", entity="小米", deleted_at=after(t))
        else:
            store.correct(
                "u",
                category="phone",
                entity="小米",
                expected_head=prior.memory_id,
                relation="neutral",
                evidence_ref="synthetic:management",
                evidence_text="合成明确取消",
                occurred_at=t.occurred_at,
                available_at=t.available_at,
            )

    client = FixedStages(data, data, on_call=change)
    with pytest.raises(CommitError, match="stale"):
        prepare(session, t, client)
    assert len(client.seen) == stage and not session.turns and not session.working_clues
    assert len(session.last_analysis_calls) == stage
    assert all(m.relation != "prefer" for m in store.list_memories("u", as_of=after(t)))


def test_change_while_selecting_view_prevents_second_call(tmp_path, monkeypatch):
    store, session = setup(tmp_path, (old(),))
    t = turn(1, "给我推荐小米")
    data = value(t, "recommend", entity="小米")
    original = store.active_memories

    def changed(*args, **kwargs):
        records = original(*args, **kwargs)
        store.delete_entity("u", category="phone", entity="小米", deleted_at=after(t))
        return records

    monkeypatch.setattr(store, "active_memories", changed)
    client = FixedStages(data, data)
    with pytest.raises(CommitError, match="stale"):
        prepare(session, t, client)
    assert len(client.seen) == 1 and not session.turns


def test_state_change_and_provider_error_still_clear_prefix(tmp_path):
    store, session = setup(tmp_path)
    t = turn(1, "合成用户原话")

    def changed(_):
        store.delete_entity("u", category="phone", entity="小米", deleted_at=after(t))

    client = FixedStages(ProviderError("transport"), on_call=changed)
    with pytest.raises(CommitError, match="stale"):
        prepare(session, t, client)
    assert not session.turns and len(session.last_analysis_calls) == 1
    assert session.last_analysis_calls[0].status == "provider_error"


def test_stale_before_call_and_before_commit_do_not_replay_deleted_text(tmp_path):
    store, session = setup(tmp_path)
    t = turn(1, "我一直喜欢三星")
    p = prepare(
        session,
        t,
        FixedStages(
            value(t, entity="三星", relation="prefer"), value(t, entity="三星", relation="prefer")
        ),
    )
    store.delete_entity("u", category="phone", entity="三星", deleted_at=after(t))
    with pytest.raises(CommitError, match="stale"):
        session.commit(p.ticket_id, as_of=after(t))
    assert not session.turns
    later = turn(2, "背景")
    store.delete_entity("u", category="phone", entity="小米", deleted_at=after(later))
    client = FixedStages(value(later, "unrelated"), value(later, "unrelated"))
    with pytest.raises(CommitError, match="stale"):
        prepare(session, later, client)
    assert not client.seen and session.last_analysis_calls == ()


@pytest.mark.parametrize(
    "outcome,expected", [("confirm", "recommend"), ("unclear", "clarify"), ("cancel", "answer")]
)
def test_real_question_and_reply_resume_original_request(tmp_path, outcome, expected):
    store, session = setup(tmp_path)
    t = turn(1, "给她推荐手机，没说她是哪位")
    p = prepare(session, t, FixedStages(ambiguous(t), ambiguous(t)))
    q = turn(2, p.memory_preview.proposal.decision.clarification, "assistant")
    session.observe_assistant(q)
    answer = turn(
        3,
        "给我自己，这次"
        if outcome == "confirm"
        else "还没决定"
        if outcome == "unclear"
        else "取消",
    )
    data = confirmation(answer, q, outcome)
    if outcome == "confirm":
        data["confirmation"]["scope"] = "current"
    client = FixedStages(data, data)
    resumed = prepare(session, answer, client)
    assert resumed.response_action == expected and len(client.seen) == 2
    assert client.seen[1]["pending_question"]["turn_id"] == q.turn_id
    assert not store.list_memories("u", as_of=after(answer))


def test_unknown_scope_confirmation_then_revision_survives_restart(tmp_path):
    store, session = setup(tmp_path, (old(),))
    t = turn(1, "我喜欢小米，但不确定只算本次还是长期")
    data = value(t, entity="小米", relation="prefer", scope="unclear")
    p = prepare(session, t, FixedStages(data, data))
    assert p.response_action == "clarify"
    q = turn(2, p.memory_preview.proposal.decision.clarification, "assistant")
    session.observe_assistant(q)
    answer = turn(3, "这是我自己的长期喜好")
    data = confirmation(answer, q)
    updated = prepare(session, answer, FixedStages(data, data))
    session.commit(updated.ticket_id, as_of=after(answer))
    reopened = JsonlMemoryStore(store.root)
    assert [m.relation for m in reopened.list_memories("u", as_of=after(answer))] == [
        "exclude",
        "prefer",
    ]
    assert [m.relation for m in reopened.active_memories("u", as_of=after(answer))] == ["prefer"]


def test_next_turn_has_only_its_own_two_call_records(tmp_path):
    _, session = setup(tmp_path)
    for i in [1, 2]:
        t = turn(i, f"背景{i}")
        p = prepare(session, t, FixedStages(value(t, "unrelated"), value(t, "unrelated")))
        assert len(p.analysis_calls) == 2 and session.last_analysis_calls == p.analysis_calls
        session.commit(p.ticket_id, as_of=after(t))


@pytest.mark.parametrize("stage", [1, 2])
def test_root_rebinding_during_call_rejects_both_stages(tmp_path, stage):
    store, session = setup(tmp_path)
    t = turn(1, "我长期喜欢三星")
    data = value(t, entity="三星", relation="prefer")

    def rebind(index):
        if index == stage:
            store.root = tmp_path / "other-root"

    client = FixedStages(data, data, on_call=rebind)
    with pytest.raises(CommitError, match="binding"):
        prepare(session, t, client)
    assert not session.turns and len(client.seen) == stage
    assert len(session.last_analysis_calls) == stage


def test_another_users_change_does_not_invalidate_current_user(tmp_path):
    store, session = setup(tmp_path)
    t = turn(1, "今天下雨")

    def unrelated_change(index):
        if index == 1:
            store.add(old(user_id="other"))

    client = FixedStages(value(t, "unrelated"), value(t, "unrelated"), on_call=unrelated_change)
    p = prepare(session, t, client)
    assert p.response_action == "continue" and len(client.seen) == 2
    assert client.seen[1]["memory_view"]["entries"] == []
    assert not store.list_memories("u", as_of=after(t))


def test_unrelated_consult_experience_then_self_preference_is_progressive(tmp_path):
    store, session = setup(tmp_path)
    user_steps = [
        (turn(1, "今天下雨"), "unrelated", None, False, None),
        (turn(3, "你怎么看三星"), "consult", "三星", False, None),
        (turn(5, "三星这台系统用着顺手"), "share_experience", "三星", True, None),
        (turn(7, "我长期更喜欢三星"), "express_preference", "三星", False, "prefer"),
    ]
    for index, (t, intent, entity, clue, relation) in enumerate(user_steps):
        data = value(t, intent, entity=entity, clue=clue, relation=relation)
        client = FixedStages(data, data)
        p = prepare(session, t, client)
        assert len(client.seen) == 2
        session.commit(p.ticket_id, as_of=after(t))
        stored = store.active_memories("u", as_of=after(t))
        assert [(m.entity, m.relation) for m in stored] == (
            [("三星", "prefer")] if relation else []
        )
        assert len(session.working_clues) == (1 if clue else 0)
        if index < 3:
            session.observe_assistant(
                turn(2 * index + 2, "系统体验不能直接代表长期品牌态度。", "assistant")
            )


def test_short_confirmation_cannot_reassign_known_self_attitude(tmp_path):
    store, session = setup(tmp_path)
    t = turn(1, "我一直喜欢OPPO，给她推荐手机，但没说她是谁")
    p = prepare(session, t, FixedStages(mixed(t), mixed(t)))
    q = turn(2, p.memory_preview.proposal.decision.clarification, "assistant")
    session.observe_assistant(q)
    t2 = turn(3, "给我妈，只算这次")
    data = confirmation(t2, q)
    data["confirmation"].update(
        subject="other", scope="current", person_reference=quote(t2, "我妈")
    )
    with pytest.raises(CommitError, match="clarification_target"):
        prepare(session, t2, FixedStages(data, data))
    assert not store.list_memories("u", as_of=after(t2))
