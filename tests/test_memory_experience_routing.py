"""体验与可执行反馈的协议边界；固定响应只验证程序行为。"""

import json

import pytest
from test_memory_extraction import subject
from test_memory_interaction import accept, after, model, prepare, turn, value
from test_memory_interaction_extraction import quote

from memrecagent.memory_commit import CommitError
from memrecagent.memory_extraction import ExtractionError
from memrecagent.memory_interaction import ConversationMemorySession
from memrecagent.memory_interaction_extraction import decode_interaction
from memrecagent.memory_store import JsonlMemoryStore


@pytest.fixture
def setup(tmp_path):
    store = JsonlMemoryStore(tmp_path / "memory")
    return store, ConversationMemorySession(store, user_id="u", session_id="s")


@pytest.mark.parametrize("entity", ["小米", "vivo"])
def test_experience_answers_without_brand_attitude_or_long_term_write(setup, entity):
    store, session = setup
    t = turn(1, f"{entity}这台用着还行，这只是单台体验")
    raw = value(t, "share_experience", entity=entity, clue=True)
    p = prepare(session, t, raw)
    assert p.response_action == "answer" and not p.memory_preview.proposal.proposed_memories
    assert not p.context_memories and not session.working_clues
    session.commit(p.ticket_id, as_of=after(t))
    assert [(c.entity, c.subject.kind) for c in session.working_clues] == [(entity, "self")]
    assert not store.list_memories("u", as_of=after(t))


def test_experience_without_named_entity_can_answer_without_invented_clue(setup):
    store, session = setup
    t = turn(1, "这台有点热")
    p = accept(session, t, value(t, "share_experience"))
    assert p.response_action == "answer" and not session.working_clues
    assert not store.list_memories("u", as_of=after(t))


def test_other_experience_remains_session_only(setup):
    store, session = setup
    t = turn(1, "我爸说小米这台有点热")
    raw = value(t, "share_experience", entity="小米", clue=True)
    raw["clues"][0]["subject"] = subject("other", [quote(t, "我爸")])
    p = accept(session, t, raw)
    assert p.response_action == "answer" and not p.context_memories
    assert session.working_clues[0].subject.kind == "other"
    assert not store.list_memories("u", as_of=after(t))


@pytest.mark.parametrize("intent", ["consult", "compare", "recommend"])
def test_experience_can_coexist_with_actual_question_or_request(setup, intent):
    _, session = setup
    t = turn(1, "小米这台发热，请比较并给我推荐手机")
    raw = value(t, intent, entity="小米", clue=True)
    raw["intents"].append("share_experience")
    p = prepare(session, t, raw)
    assert (
        p.response_action
        == {"consult": "answer", "compare": "compare", "recommend": "recommend"}[intent]
    )


@pytest.mark.parametrize("extra_experience", [False, True])
def test_explicit_feedback_with_clue_still_requires_target(setup, extra_experience):
    store, session = setup
    t = turn(1, "小米这台发热，别再推这款了")
    raw = value(t, "feedback", entity="小米", clue=True)
    if extra_experience:
        raw["intents"].append("share_experience")
    p = prepare(session, t, raw)
    assert p.response_action == "feedback_target_required"
    assert not p.memory_preview.proposal.proposed_memories
    assert not store.list_memories("u", as_of=after(t))


def test_bare_feedback_still_requires_target(setup):
    _, session = setup
    t = turn(1, "不感兴趣")
    assert prepare(session, t, value(t, "feedback")).response_action == "feedback_target_required"


def test_stable_attitude_is_only_saved_when_separately_expressed(setup):
    store, session = setup
    t = turn(1, "小米这台发热，我一直排除小米")
    raw = value(t, entity="小米", relation="exclude")
    raw["intents"].append("share_experience")
    raw["topics"] = [quote(t, "小米这台")]
    p = prepare(session, t, raw)
    assert not store.list_memories("u", as_of=after(t))
    session.commit(p.ticket_id, as_of=after(t))
    assert [(m.entity, m.relation) for m in store.active_memories("u", as_of=after(t))] == [
        ("小米", "exclude")
    ]


def test_unclear_scope_still_prevents_experience_round_commit(setup):
    store, session = setup
    t = turn(1, "小米这台还行，我喜欢小米但范围未定")
    raw = value(t, entity="小米", relation="prefer", scope="unclear")
    raw["intents"].append("share_experience")
    raw["topics"] = [quote(t, "小米这台")]
    p = prepare(session, t, raw)
    assert p.response_action == "clarify"
    with pytest.raises(CommitError, match="confirmation_required"):
        session.commit(p.ticket_id, as_of=after(t))
    assert not store.list_memories("u", as_of=after(t))


def test_experience_does_not_relax_entity_evidence_validation(setup):
    store, session = setup
    t = turn(1, "小米这台发热")
    raw = value(t, "share_experience", entity="小米", clue=True)
    raw["clues"][0]["entity"] = "三星"
    with pytest.raises(ExtractionError, match="entity_evidence"):
        prepare(session, t, raw)
    assert not store.list_memories("u", as_of=after(t))


def test_unknown_experience_enum_is_not_silently_accepted():
    t = turn(1, "小米这台发热")
    raw = value(t, "share_experiences")
    with pytest.raises(ExtractionError, match="interaction_intents"):
        decode_interaction(json.dumps(raw), turns=(t,))


def test_model_failure_still_invalidates_previous_experience_ticket(setup):
    _, session = setup
    t = turn(1, "小米这台发热")
    p = prepare(session, t, value(t, "share_experience", entity="小米", clue=True))
    with pytest.raises(ExtractionError):
        session.prepare_user(turn(2, "继续"), complete=model({}), as_of=after(turn(2, "继续")))
    with pytest.raises(CommitError, match="ticket"):
        session.commit(p.ticket_id, as_of=after(turn(2, "继续")))
    assert not session.working_clues


def test_existing_exclusion_is_not_activated_or_revised_by_plain_experience(setup):
    store, session = setup
    original = turn(1, "我一直排除小米")
    accept(session, original, value(original, entity="小米", relation="exclude"))
    before = store.snapshot_token("u")
    t = turn(2, "小米这台用着还行，但这只是单台体验")
    p = prepare(session, t, value(t, "share_experience", entity="小米", clue=True))
    assert p.response_action == "answer" and not p.context_memories
    session.commit(p.ticket_id, as_of=after(t))
    assert store.snapshot_token("u") == before
    assert [(m.entity, m.relation) for m in store.active_memories("u", as_of=after(t))] == [
        ("小米", "exclude")
    ]
    assert len(session.working_clues) == 1
    later = turn(3, "你怎么看小米")
    background = prepare(session, later, value(later, entity="小米"))
    assert [(m.entity, m.relation) for m in background.context_memories] == [("小米", "exclude")]
