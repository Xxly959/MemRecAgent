"""原文候选不被改写，规范键贯通提案、会话与目录。"""

from dataclasses import replace

from test_conversation_products import process
from test_entity_aliases import document
from test_memory_interaction import after, turn, value
from test_request_attributes import host, plain

from memrecagent.entity_aliases import install_entity_aliases
from memrecagent.memory_store import JsonlMemoryStore


def configured(root, session="s"):
    if not (root / "memory/entity-aliases.json").exists():
        install_entity_aliases(root / "memory", document())
    return host(root, session)


def test_raw_alias_quotes_save_and_revision_remain_original_but_share_head(tmp_path):
    h = configured(tmp_path)
    first = turn(1, "以后别给我推荐BrandA，请长期保存。")
    process(h, first, value(first, entity="BrandA", relation="exclude"))
    h.observe_assistant(turn(2, "已记录", "assistant"))
    second = turn(3, "改为以后优先A牌，请更新我的长期偏好。")
    process(h, second, value(second, entity="A牌", relation="prefer"))
    store = JsonlMemoryStore(tmp_path / "memory")
    history = store.list_memories("u", as_of=after(second))
    assert len(history) == 2 and history[1].supersedes == history[0].memory_id
    assert [m.entity for m in history] == ["品牌A", "品牌A"]
    assert history[0].evidence_text == first.content and history[1].evidence_text == second.content
    t = turn(5, "给我推荐手机")
    ranked = process(configured(tmp_path, "new"), t, plain(t))["ranking"]
    assert ranked["items"][0]["item_id"] == "A"


def test_alias_current_request_matches_catalog_alias_without_mutating_memory(tmp_path):
    h = configured(tmp_path)
    # 可信目录原始别名应由ranker构建时规范化，使用真实构造器再次绑定。
    from memrecagent.memory_products import MemoryProductRecommender

    facts = tuple(replace(f, entity="BrandA") if f.item_id == "A" else f for f in h._ranker._facts)
    ranker = MemoryProductRecommender(
        h._service, h._memory._store, facts, facts_bundle_id=h._service.bundle_id
    )
    from test_memory_products import request

    result = ranker.recommend(**request(requested_entity="A牌"))
    assert [p["item_id"] for p in result["items"]] == ["A"]
    assert result["items"][0]["entity"] == "品牌A"
    assert result["items"][0]["entity_alias_source"] == "synthetic:catalog:v1"


def test_alias_consult_activates_canonical_memory_for_final_analyst_and_reply(tmp_path):
    from test_memory_analyst_host import FixedStages
    from test_memory_interaction_extraction import quote

    h = configured(tmp_path)
    t = turn(1, "以后我喜欢BrandA，请保存。")
    process(h, t, value(t, entity="BrandA", relation="prefer"))
    h.observe_assistant(turn(2, "已保存", "assistant"))
    q = turn(3, "给我说说A牌手机")
    raw = plain(q, "consult")
    raw["topics"] = [quote(q, "A牌手机")]
    from test_memory_extraction import subject

    raw["candidate"]["request"]["beneficiary"] = subject()
    client = FixedStages(raw, raw)
    result = h.process_user(q, complete=client, as_of=after(q), strategy="popularity", k=3)
    final_input = client.seen[1]
    assert final_input["memory_view"]["entries"][0]["entity"] == "品牌A"
    assert result["context_memories"][0]["entity"] == "品牌A"
