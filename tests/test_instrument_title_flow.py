"""实际宿主消费完整商品标题，规范化只发生在键上，不改原文证据。"""

from datetime import UTC, datetime, timedelta

import pytest
from test_conversation_domain import DOMAIN, raw
from test_memory_analyst_host import FixedStages
from test_memory_extraction import subject
from test_product_service import service

from memrecagent.conversation_products import ConversationProductSession
from memrecagent.entity_aliases import install_entity_aliases
from memrecagent.instrument_catalog import build_instrument_mapping, build_instrument_title_aliases
from memrecagent.memory_extraction import Turn
from memrecagent.memory_products import decode_product_entity_fact
from memrecagent.memory_store import JsonlMemoryStore
from memrecagent.product_bundle import load_bundle, save_bundle


@pytest.mark.parametrize("mode", ["direct", "analyst", "analyst_blind"])
def test_title_and_native_id_share_revision_rank_and_consult_head(tmp_path, mode):
    base = service()
    base.counts = dict(A=10, B=9, C=8)
    base.seen = {}
    base.products = {
        item: dict(title=title, categories=["Musical Instruments"])
        for item, title in zip(
            "ABC", ["Studio Guitar", "Travel Drums", "Concert Harp"], strict=True
        )
    }
    bundle = tmp_path / "bundle"
    save_bundle(base, bundle, provenance=dict(kind="synthetic"))
    mapping = build_instrument_mapping(bundle)
    aliases, _ = build_instrument_title_aliases(bundle)
    root = tmp_path / "memory"
    install_entity_aliases(root, aliases)
    facts = tuple(decode_product_entity_fact(row) for row in mapping["facts"])
    native = load_bundle(bundle)

    def host(sid):
        return ConversationProductSession(
            native,
            JsonlMemoryStore(root),
            facts,
            facts_bundle_id=native.bundle_id,
            user_id="u",
            session_id=sid,
            category=DOMAIN,
        )

    now = datetime.now(UTC)

    def run(h, i, text, intent="consult", entity=None, relation=None):
        at = now + timedelta(seconds=i)
        t = Turn(f"T{i}", "user", text, at, at)
        candidate = raw(t, intent, entity=entity, relation=relation)
        if intent == "consult":
            candidate["candidate"]["request"]["beneficiary"] = subject()
        client = FixedStages(candidate, candidate)
        result = h.process_user(
            t,
            complete=client,
            as_of=at + timedelta(microseconds=1),
            analysis_mode=mode,
            strategy="popularity",
            k=3,
        )
        return result, client, t

    h = host("s1")
    first, _, t1 = run(
        h,
        1,
        "以后请排除Studio Guitar，作为我的长期偏好。",
        entity="Studio Guitar",
        relation="exclude",
    )
    assert len(first["new_memory_ids"]) == 1
    reopened = host("s2")
    ranked, _, _ = run(reopened, 2, "给我推荐三款乐器", "recommend")
    assert [p["item_id"] for p in ranked["ranking"]["items"]] == ["B", "C"]
    consult, client, _ = run(reopened, 3, "你怎么看Studio Guitar？", entity="Studio Guitar")
    assert consult["context_memories"][0]["entity"] == "A"
    if mode != "direct":
        assert client.seen[1]["memory_view"]["entries"][0]["entity"] == "A"
    files = list(root.glob("*.jsonl"))
    before = {p: p.read_bytes() for p in files}
    override, _, _ = run(
        reopened, 4, "这次只看Studio Guitar，长期偏好别改。", "recommend", entity="Studio Guitar"
    )
    assert [p["item_id"] for p in override["ranking"]["items"]] == ["A"]
    assert {p: p.read_bytes() for p in files} == before
    _, _, t5 = run(
        reopened, 5, "以后改为优先商品A，请更新我的长期偏好。", entity="A", relation="prefer"
    )
    history = JsonlMemoryStore(root).list_memories(
        "u", as_of=t5.available_at + timedelta(seconds=1)
    )
    assert [m.entity for m in history] == ["A", "A"]
    assert history[1].supersedes == history[0].memory_id
    assert history[0].evidence_text == t1.content
    assert history[1].evidence_text == t5.content
