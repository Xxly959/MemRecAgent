"""确定性话题对齐的真实宿主回归；固定响应不代表模型理解效果。"""

from datetime import timedelta

import pytest
from test_memory_extraction import NOW, subject
from test_memory_interaction import after, prepare, turn, value
from test_memory_interaction_extraction import quote

from memrecagent.memory import PreferenceMemory
from memrecagent.memory_interaction import ConversationMemorySession
from memrecagent.memory_store import JsonlMemoryStore


def seed(
    store,
    entity="荣耀",
    *,
    user_id="u",
    category="phone",
    relation="prefer",
    occurred_at=NOW,
    available_at=NOW,
):
    memory = PreferenceMemory(
        memory_id=f"{user_id}-{category}-{entity}",
        user_id=user_id,
        category=category,
        entity=entity,
        relation=relation,
        evidence_ref="synthetic:topic-alignment",
        evidence_text=f"合成证据：一直喜欢{entity}",
        occurred_at=occurred_at,
        available_at=available_at,
    )
    store.add(memory)
    return memory


@pytest.fixture
def setup(tmp_path):
    store = JsonlMemoryStore(tmp_path / "memory")
    return store, lambda **kwargs: ConversationMemorySession(
        store, user_id="u", session_id="s", **kwargs
    )


def consult(session, topic, *, intent="consult", other=False):
    t = turn(10, f"给我爸说说{topic}" if other else f"说说{topic}")
    raw = value(t, intent)
    raw["topics"] = [quote(t, topic)]
    if other:
        raw["candidate"]["request"]["beneficiary"] = subject("other", [quote(t, "我爸")])
    return t, raw, prepare(session, t, raw)


@pytest.mark.parametrize("intent", ["consult", "compare"])
@pytest.mark.parametrize("entity", ["荣耀", "三星", "甲牌"])
def test_phone_phrase_reads_only_matching_memory_without_changing_raw_topic(setup, intent, entity):
    store, make_session = setup
    expected = seed(store, entity)
    seed(store, "其他品牌")
    session = make_session()
    before = store.snapshot_token("u")
    _, _, preview = consult(session, f"{entity}手机", intent=intent)
    assert preview.context_memories == (expected,)
    assert preview.topic_entities == (f"{entity}手机",)
    assert preview.response_action == ("compare" if intent == "compare" else "answer")
    session.commit(preview.ticket_id, as_of=after(turn(10, "接受")))
    assert store.snapshot_token("u") == before


@pytest.mark.parametrize(
    "topic", ["荣耀手机壳", "新荣耀手机", "荣耀手机新品", "荣", "HONOR手机", "苹果手机"]
)
def test_partial_or_unknown_topic_does_not_guess_existing_brand(setup, topic):
    store, make_session = setup
    seed(store)
    assert not consult(make_session(), topic)[2].context_memories


def test_exact_entity_wins_over_suffix_interpretation(setup):
    store, make_session = setup
    seed(store, "荣耀")
    exact = seed(store, "荣耀手机")
    assert consult(make_session(), "荣耀手机")[2].context_memories == (exact,)


def test_complete_entity_does_not_activate_shorter_overlapping_name(setup):
    store, make_session = setup
    seed(store, "荣")
    expected = seed(store, "荣耀")
    assert consult(make_session(), "荣耀手机")[2].context_memories == (expected,)


def test_comparison_resolves_each_topic_and_deduplicates_background(setup):
    store, make_session = setup
    a, b = seed(store), seed(store, "三星")
    t = turn(10, "荣耀手机和三星手机有什么区别")
    raw = value(t, "compare")
    raw["topics"] = [quote(t, "荣耀手机"), quote(t, "荣耀"), quote(t, "三星手机")]
    p = prepare(make_session(), t, raw)
    assert set(p.context_memories) == {a, b} and len(p.context_memories) == 2
    assert set(p.topic_entities) == {"荣耀手机", "荣耀", "三星手机"}


@pytest.mark.parametrize(
    "kind",
    [
        "other_user",
        "category",
        "future",
        "late_available",
        "expired",
        "neutral",
        "other_beneficiary",
    ],
)
def test_topic_resolution_still_obeys_visible_memory_boundaries(setup, kind):
    store, make_session = setup
    kwargs = {}
    session_kwargs = {}
    if kind == "other_user":
        kwargs["user_id"] = "someone-else"
    elif kind == "category":
        kwargs["category"] = "laptop"
    elif kind == "future":
        kwargs.update(occurred_at=NOW + timedelta(hours=1), available_at=NOW + timedelta(hours=1))
    elif kind == "late_available":
        kwargs["available_at"] = NOW + timedelta(hours=1)
    elif kind == "expired":
        session_kwargs["max_memory_age"] = timedelta(minutes=5)
    elif kind == "neutral":
        kwargs["relation"] = "neutral"
    seed(store, **kwargs)
    assert not consult(
        make_session(**session_kwargs), "荣耀手机", other=kind == "other_beneficiary"
    )[2].context_memories


def test_deleted_brand_is_not_reintroduced_by_topic_suffix(setup):
    store, make_session = setup
    seed(store)
    store.delete_entity("u", category="phone", entity="荣耀", deleted_at=NOW + timedelta(minutes=1))
    # 外部变更后按既有契约建立新的工作会话。
    session = ConversationMemorySession(store, user_id="u", session_id="fresh")
    assert not consult(session, "荣耀手机")[2].context_memories
