"""引用与请求恢复的实际宿主回归；固定模型返回只验证工程行为。"""

from copy import deepcopy
from dataclasses import replace

import pytest
from test_conversation_products import process
from test_memory_extraction import subject
from test_memory_interaction import after, prepare, turn, value
from test_memory_interaction_extraction import decode, envelope, quote
from test_memory_topic_alignment import seed
from test_request_attributes import host, ids, payload, plain

from memrecagent.memory_commit import CommitError
from memrecagent.memory_extraction import ExtractionError
from memrecagent.memory_interaction import ConversationMemorySession
from memrecagent.memory_store import JsonlMemoryStore


def test_repeated_topic_does_not_block_unique_preference_evidence(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    session = ConversationMemorySession(store, user_id="u", session_id="s")
    t = turn(1, "荣耀我一直喜欢，换机还是优先荣耀。")
    raw = value(t, entity="荣耀", relation="prefer")
    preview = prepare(session, t, raw)
    assert not store.list_memories("u", as_of=after(t))
    session.commit(preview.ticket_id, as_of=after(t))
    memories = store.active_memories("u", as_of=after(t))
    assert [(m.entity, m.relation) for m in memories] == [("荣耀", "prefer")]
    assert memories[0].evidence_text == t.content


def test_repeated_consult_topics_read_matching_memory_without_creating_preference(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    expected = seed(store, "荣耀")
    seed(store, "苹果")
    session = ConversationMemorySession(store, user_id="u", session_id="s")
    before = store.snapshot_token("u")
    t = turn(1, "三星和荣耀怎么选？我想知道荣耀与三星差在哪。")
    raw = value(t, "compare")
    raw["topics"] = [quote(t, "荣耀"), quote(t, "三星"), quote(t, "荣耀")]
    preview = prepare(session, t, raw)
    assert preview.topic_entities == ("三星", "荣耀")
    assert preview.context_memories == (expected,)
    session.commit(preview.ticket_id, as_of=after(t))
    assert store.snapshot_token("u") == before


@pytest.mark.parametrize("kind", ["assertion", "person", "request", "clue"])
def test_topic_repetition_does_not_relax_binding_evidence(kind):
    t = turn(1, "我爸喜欢荣耀，我爸喜欢荣耀。")
    raw = value(t, entity="荣耀", relation="prefer")
    assertion = raw["candidate"]["assertions"][0]
    if kind == "assertion":
        assertion["evidence"] = [quote(t, "喜欢荣耀")]
    elif kind == "person":
        assertion["subject"] = subject("other", [quote(t, "我爸")])
    elif kind == "request":
        raw["intents"].append("recommend")
        raw["candidate"]["request"].update(
            is_recommendation=True,
            beneficiary=subject(),
            category="phone",
            evidence=[quote(t, "荣耀")],
        )
    else:
        raw["clues"] = [
            dict(subject=subject(), category="phone", entity="荣耀", evidence=[quote(t, "荣耀")])
        ]
    with pytest.raises(ExtractionError, match="quote_ambiguous"):
        decode(raw, (t,))


@pytest.mark.parametrize(
    "kind,code",
    [
        ("missing", "quote_missing"),
        ("assistant", "quote_role"),
        ("empty", "quote_missing"),
        ("type", "quote_missing"),
    ],
)
def test_topic_still_requires_exact_user_source(kind, code):
    old, t = turn(1, "荣耀荣耀"), turn(2, "咨询手机")
    raw = envelope()
    raw["topics"] = [quote(old, "荣耀")]
    if kind == "missing":
        raw["topics"][0]["turn_id"] = t.turn_id
    elif kind == "assistant":
        old = replace(old, role="assistant")
    elif kind == "empty":
        raw["topics"][0]["quote"] = " "
    else:
        raw["topics"][0]["quote"] = {"forged": True}
    with pytest.raises(ExtractionError, match=code):
        decode(deepcopy(raw), (old, t))


def test_complete_update_without_current_request_starts_new_temporary_budget(tmp_path):
    h = host(tmp_path)
    t = turn(1, "这次我的预算调到两千八，给我推荐手机。")
    result = process(h, t, payload(t, op="update", amount=2800, recommend=True))
    assert ids(result) == ["B", "C"]
    assert result["new_memory_ids"] == []
    assert not JsonlMemoryStore(tmp_path / "memory").list_memories("u", as_of=after(t))
    fresh = host(tmp_path, "fresh")
    t = turn(3, "给我推荐手机。")
    assert ids(process(fresh, t, plain(t))) == ["A", "B", "C"]


def test_update_after_end_uses_new_explicit_value_without_old_request_id(tmp_path):
    h = host(tmp_path)
    t = turn(1, "这次预算两千一。")
    process(h, t, payload(t, amount=2100))
    old_id = h._attribute_request.request_id
    t = turn(3, "这次选购结束。")
    process(h, t, payload(t, op="end", amount=None))
    t = turn(5, "我的预算调整到两千八，给我推荐手机。")
    result = process(h, t, payload(t, op="update", amount=2800, recommend=True))
    assert ids(result) == ["B", "C"]
    assert h._attribute_request.request_id != old_id
    assert [c.value for c in h._attribute_request.conditions] == [2800]


@pytest.mark.parametrize("amount", [None, 2800])
def test_missing_request_cannot_turn_removal_into_unapproved_relaxation(tmp_path, amount):
    h = host(tmp_path)
    t = turn(1, "这次不限制重量，预算两千八。")
    raw = payload(t, op="update", amount=amount, remove=("weight_g",))
    with pytest.raises(CommitError, match="attribute_request_missing"):
        process(h, t, raw)
    assert h._attribute_request is None
    assert not JsonlMemoryStore(tmp_path / "memory").list_memories("u", as_of=after(t))


def test_update_cannot_switch_person_and_carry_previous_budget(tmp_path):
    h = host(tmp_path)
    t = turn(1, "这次我预算三千。")
    process(h, t, payload(t))
    t = turn(3, "我叔叔预算调到四千五。")
    raw = payload(t, op="update", amount=4500, who=subject("other", [quote(t, "我叔叔")]))
    with pytest.raises(CommitError, match="attribute_request_missing"):
        process(h, t, raw)
    t = turn(5, "仍然给我自己推荐手机。")
    assert ids(process(h, t, plain(t))) == ["B", "C"]


def test_new_request_recovery_is_not_published_before_successful_commit(tmp_path, monkeypatch):
    h = host(tmp_path)
    original = h._memory.commit

    def fail(*args, **kwargs):
        raise CommitError("stale")

    monkeypatch.setattr(h._memory, "commit", fail)
    t = turn(1, "这次我的预算调到两千八。")
    with pytest.raises(CommitError, match="stale"):
        process(h, t, payload(t, op="update", amount=2800))
    assert h._attribute_request is None
    monkeypatch.setattr(h._memory, "commit", original)
    t = turn(3, "给我推荐手机。")
    assert ids(process(h, t, plain(t))) == ["A", "B", "C"]
