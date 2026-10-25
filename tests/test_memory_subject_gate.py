"""显式人物未解必须先澄清，不能发布或改派同轮明确态度。"""

import pytest
from test_memory_extraction import subject
from test_memory_interaction import accept, after, confirmation, prepare, turn, value
from test_memory_interaction_extraction import quote

from memrecagent.memory_commit import CommitError
from memrecagent.memory_interaction import ConversationMemorySession
from memrecagent.memory_store import JsonlMemoryStore


@pytest.fixture
def setup(tmp_path):
    store = JsonlMemoryStore(tmp_path / "memory")
    return store, ConversationMemorySession(store, user_id="u", session_id="s")


def ambiguous(t):
    raw = value(t, "recommend")
    raw["candidate"]["request"]["beneficiary"] = subject("unknown")
    raw["candidate"]["unresolved"] = [{"field": "subject", "evidence": [quote(t)]}]
    return raw


@pytest.mark.parametrize("existing", [False, True])
def test_explicit_unknown_recipient_clarifies_even_without_preference_conflict(setup, existing):
    store, session = setup
    if existing:
        old = turn(1, "我一直排除索尼")
        accept(session, old, value(old, entity="索尼", relation="exclude"))
    before = store.snapshot_token("u")
    t = turn(2, "给她推荐手机，但我还没说她是哪位")
    p = prepare(session, t, ambiguous(t))
    assert p.response_action == "clarify"
    assert not p.memory_preview.proposal.proposed_memories
    with pytest.raises(CommitError, match="confirmation_required"):
        session.commit(p.ticket_id, as_of=after(t))
    assert store.snapshot_token("u") == before


def mixed(t):
    raw = ambiguous(t)
    raw["intents"].append("express_preference")
    raw["candidate"]["assertions"] = value(t, entity="OPPO", relation="prefer")["candidate"][
        "assertions"
    ]
    raw["candidate"]["assertions"][0]["evidence"] = [quote(t, "我一直喜欢OPPO")]
    raw["candidate"]["unresolved"][0]["evidence"] = [quote(t, "给她推荐手机，但没说她是谁")]
    return raw


def test_known_attitude_is_not_published_before_unknown_request_recipient_is_resolved(setup):
    store, session = setup
    t = turn(1, "我一直喜欢OPPO，给她推荐手机，但没说她是谁")
    p = prepare(session, t, mixed(t))
    assert p.response_action == "clarify"
    assert not p.memory_preview.proposal.proposed_memories
    with pytest.raises(CommitError, match="confirmation_required"):
        session.commit(p.ticket_id, as_of=after(t))
    assert not store.list_memories("u", as_of=after(t))


def test_explicit_unknown_attitude_subject_clarifies_without_old_memory_conflict(setup):
    store, session = setup
    t = turn(1, "喜欢OPPO，但我没有说是谁喜欢")
    raw = value(t, entity="OPPO", relation="prefer")
    raw["candidate"]["assertions"][0]["subject"] = subject("unknown")
    raw["candidate"]["unresolved"] = [{"field": "subject", "evidence": [quote(t)]}]
    p = prepare(session, t, raw)
    assert p.response_action == "clarify" and not p.memory_preview.proposal.proposed_memories
    assert not store.list_memories("u", as_of=after(t))


def test_unknown_recipient_without_explicit_unresolved_keeps_generic_recommendation(setup):
    _, session = setup
    t = turn(1, "推荐一些手机")
    raw = ambiguous(t)
    raw["candidate"]["unresolved"] = []
    p = prepare(session, t, raw)
    assert p.response_action == "recommend" and not p.context_memories


def test_actual_reply_can_resolve_request_only_recipient(setup):
    store, session = setup
    t = turn(1, "给她推荐手机，还没说她是谁")
    p = prepare(session, t, ambiguous(t))
    assert p.response_action == "clarify"
    q = turn(2, p.memory_preview.proposal.decision.clarification, "assistant")
    session.observe_assistant(q)
    reply = turn(3, "是给我自己，这次推荐")
    raw = confirmation(reply, q)
    raw["confirmation"]["scope"] = "current"
    accepted = accept(session, reply, raw)
    assert accepted.response_action == "recommend"
    assert accepted.memory_preview.proposal.decision.beneficiary_id == "self"
    assert not store.list_memories("u", as_of=after(reply))


def test_short_recipient_confirmation_cannot_reassign_known_attitude(setup):
    store, session = setup
    t = turn(1, "我一直喜欢OPPO，给她推荐手机，但没说她是谁")
    p = prepare(session, t, mixed(t))
    assert p.response_action == "clarify"
    q = turn(2, p.memory_preview.proposal.decision.clarification, "assistant")
    session.observe_assistant(q)
    reply = turn(3, "是我妈，只算这次")
    raw = confirmation(reply, q)
    raw["confirmation"].update(
        subject="other", scope="current", person_reference=quote(reply, "我妈")
    )
    with pytest.raises(CommitError, match="clarification_target"):
        prepare(session, reply, raw)
    assert not store.list_memories("u", as_of=after(reply))


def test_unclear_reply_keeps_original_request_until_actual_resolution(setup):
    _, session = setup
    t = turn(1, "给她推荐手机，但她是哪位还不明确")
    p = prepare(session, t, ambiguous(t))
    q = turn(2, p.memory_preview.proposal.decision.clarification, "assistant")
    session.observe_assistant(q)
    reply = turn(3, "还没想好")
    still = prepare(session, reply, confirmation(reply, q, "unclear"))
    assert still.response_action == "clarify"
    q2 = turn(4, still.memory_preview.proposal.decision.clarification, "assistant")
    session.observe_assistant(q2)
    reply2 = turn(5, "给我自己，这次")
    raw = confirmation(reply2, q2)
    raw["confirmation"]["scope"] = "current"
    assert prepare(session, reply2, raw).response_action == "recommend"


def test_cancelled_request_is_not_resumed_by_confirmation(setup):
    store, session = setup
    t = turn(1, "给她推荐手机，人物不明确")
    p = prepare(session, t, ambiguous(t))
    q = turn(2, p.memory_preview.proposal.decision.clarification, "assistant")
    session.observe_assistant(q)
    reply = turn(3, "算了，不用推荐")
    cancelled = accept(session, reply, confirmation(reply, q, "cancel"))
    assert cancelled.response_action == "answer"
    assert not store.list_memories("u", as_of=after(reply))


def test_resolved_current_request_uses_original_policy_background(setup):
    _, session = setup
    old = turn(1, "我长期排除索尼")
    accept(session, old, value(old, entity="索尼", relation="exclude"))
    t = turn(2, "这次给她看索尼手机，还没说她是谁")
    raw = ambiguous(t)
    raw["topics"] = [quote(t, "索尼")]
    raw["candidate"]["request"]["requested_entity"] = "索尼"
    p = prepare(session, t, raw)
    q = turn(3, p.memory_preview.proposal.decision.clarification, "assistant")
    session.observe_assistant(q)
    reply = turn(4, "是给我自己，只算这次")
    raw = confirmation(reply, q)
    raw["confirmation"]["scope"] = "current"
    resumed = prepare(session, reply, raw)
    assert resumed.response_action == "recommend" and not resumed.context_memories
