"""宿主品类绑定贯穿实际对话与重启排名；固定响应不证明语言理解。"""

from dataclasses import replace

import pytest
from test_memory_analyst_host import FixedStages
from test_memory_extraction import subject
from test_memory_interaction import after, turn, value
from test_memory_interaction_extraction import quote
from test_memory_products import engine
from test_request_attributes import payload

from memrecagent.conversation_products import ConversationProductSession
from memrecagent.memory_commit import CommitError
from memrecagent.memory_store import JsonlMemoryStore

DOMAIN = "musical_instruments"


def session(path, sid="s"):
    base = engine(path / "memory")
    facts = tuple(replace(f, category=DOMAIN, entity=f.item_id) for f in base._facts)
    return ConversationProductSession(
        base._service,
        JsonlMemoryStore(path / "memory"),
        facts,
        facts_bundle_id=base._service.bundle_id,
        user_id="u",
        session_id=sid,
        category=DOMAIN,
    )


def raw(t, *args, **kwargs):
    result = value(t, *args, **kwargs)
    req = result["candidate"]["request"]
    if req["category"] is not None:
        req["category"] = DOMAIN
    for row in result["candidate"]["assertions"] + result["clues"]:
        row["category"] = DOMAIN
    return result


def process(host, t, result, mode="analyst"):
    client = FixedStages(result, result)
    output = host.process_user(
        t,
        complete=client,
        as_of=after(t),
        strategy="popularity",
        k=3,
        analysis_mode=mode,
    )
    return output, client


@pytest.mark.parametrize("mode", ["direct", "analyst", "analyst_blind"])
def test_instrument_memory_survives_restart_and_changes_actual_rank(tmp_path, mode):
    host = session(tmp_path)
    t = turn(1, "我长期排除商品A。")
    saved, _ = process(host, t, raw(t, entity="A", relation="exclude"), mode)
    assert len(saved["new_memory_ids"]) == 1
    reopened = session(tmp_path, "new")
    t2 = turn(3, "给我推荐乐器")
    result, client = process(reopened, t2, raw(t2, "recommend"), mode)
    assert [p["item_id"] for p in result["ranking"]["items"]] == ["B", "C"]
    assert all(m["output_contract"]["catalog_category"] == DOMAIN for m in client.seen)
    stored = JsonlMemoryStore(tmp_path / "memory").list_memories("u", as_of=after(t2))
    assert len(stored) == 1 and stored[0].category == DOMAIN


def test_instrument_consult_loads_exact_entity_into_analyst_view(tmp_path):
    host = session(tmp_path)
    t = turn(1, "我长期排除商品A。")
    process(host, t, raw(t, entity="A", relation="exclude"))
    reopened = session(tmp_path, "new")
    t2 = turn(3, "你怎么看商品A？")
    candidate = raw(t2, entity="A")
    candidate["candidate"]["request"]["beneficiary"] = {"kind": "self", "references": []}
    result, client = process(reopened, t2, candidate)
    assert client.seen[1]["memory_view"]["entries"][0]["category"] == DOMAIN
    assert result["context_memories"][0]["category"] == DOMAIN


@pytest.mark.parametrize("mode", ["direct", "analyst", "analyst_blind"])
def test_wrong_category_cannot_be_committed(tmp_path, mode):
    host = session(tmp_path)
    t = turn(1, "我长期排除商品A。")
    with pytest.raises(CommitError, match="category"):
        process(host, t, value(t, entity="A", relation="exclude"), mode)
    assert JsonlMemoryStore(tmp_path / "memory").list_memories("u", as_of=after(t)) == ()


def test_phone_attribute_protocol_is_not_silently_applied_to_instruments(tmp_path):
    host = session(tmp_path)
    t = turn(1, "这次预算3000元")
    with pytest.raises(CommitError, match="category_attributes"):
        process(host, t, payload(t))
    assert JsonlMemoryStore(tmp_path / "memory").list_memories("u", as_of=after(t)) == ()


def test_instrument_other_person_does_not_inherit_owner_exclusion(tmp_path):
    host = session(tmp_path)
    t = turn(1, "我长期排除商品A。")
    process(host, t, raw(t, entity="A", relation="exclude"))
    t2 = turn(3, "给我叔叔推荐乐器")
    candidate = raw(t2, "recommend")
    candidate["candidate"]["request"]["beneficiary"] = subject("other", [quote(t2, "我叔叔")])
    result, _ = process(host, t2, candidate)
    assert [p["item_id"] for p in result["ranking"]["items"]] == ["A", "B", "C"]


def test_host_rejects_category_absent_from_facts_before_any_call(tmp_path):
    base = engine(tmp_path / "memory")
    with pytest.raises(ValueError, match="category"):
        ConversationProductSession(
            base._service,
            JsonlMemoryStore(tmp_path / "memory"),
            base._facts,
            facts_bundle_id=base._service.bundle_id,
            user_id="u",
            session_id="s",
            category=DOMAIN,
        )
