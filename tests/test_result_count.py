"""本轮数量有原文位置、不落长期、不从预算或过去购买猜数量。"""

import json

import pytest
from test_conversation_products import process, session
from test_memory_analyst_host import FixedStages
from test_memory_interaction import after, turn, value
from test_memory_interaction_extraction import quote
from test_request_attributes import host, ids, payload

from memrecagent.memory_analyst import AnalystError
from memrecagent.memory_extraction import ExtractionError
from memrecagent.memory_interaction_extraction import decode_interaction
from memrecagent.memory_store import JsonlMemoryStore
from memrecagent.result_count import resolve_result_count


@pytest.mark.parametrize(
    "text,want",
    [
        ("给我推荐两款手机", 2),
        ("再给三款", 3),
        ("改为5款，推荐手机", 5),
        ("手机推荐十部", 10),
        ("推荐手机，3款就行", 3),
        ("推荐两款，两款就好", 2),
        ("我之前买了三台手机，现在推荐2款", 2),
        ("不要三款，推荐两款", 2),
        ("最多推荐四款", 4),
        ("别超过3款", 3),
        ("请推荐２款手机", 2),
        ("我只有三台手机，给我推荐手机", None),
        ("上次推荐三款，这次推荐手机", None),
        ("预算三千，推荐手机", None),
        ("推荐3GB内存手机", None),
    ],
)
def test_clear_count_or_noncount_literal_positions(text, want):
    result = resolve_result_count(text)
    assert result.count == want and result.clarification is None
    if want is not None:
        assert result.spans and all(0 <= start < end <= len(text) for start, end in result.spans)


@pytest.mark.parametrize(
    "text",
    [
        "推荐两三款",
        "推荐2到3款",
        "推荐两款或三款",
        "推荐11款",
        "推荐0款",
        "推荐1.5款",
        "文案写着“推荐三款”，给我推荐手机",
    ],
)
def test_ambiguous_or_out_of_bound_quantity_requires_explicit_retry(text):
    result = resolve_result_count(text)
    assert result.count is None and result.clarification


def test_quote_or_past_count_does_not_override_current_request():
    text = "商家文案说“推荐三款”，这次给我推荐两款手机"
    result = resolve_result_count(text)
    assert result.count == 2
    assert [text[a:b] for a, b in result.spans] == ["两款"]


def test_count_follows_current_turn_and_four_variants_share_accepted_count(tmp_path):
    h = session(tmp_path)
    t = turn(1, "我长期喜欢品牌C")
    process(h, t, value(t, entity="品牌C", relation="prefer"))
    t = turn(3, "给我推荐两款手机")
    result = process(h, t, value(t, "recommend"))
    assert len(result["ranking"]["items"]) == 2 and result["ranking"]["requested_count"] == 2
    assert h._last_request["k"] == 2
    store = JsonlMemoryStore(tmp_path / "memory")
    before = store.snapshot_token("u")
    for mem, strategy in [
        (False, "popularity"),
        (True, "popularity"),
        (False, "bpr_cold_popularity"),
        (True, "bpr_cold_popularity"),
    ]:
        r = h.rank_current(use_long_term_memory=mem, strategy=strategy)
        assert len(r["items"]) == 2 and r["requested_count"] == 2
        assert r["count_evidence"] == result["ranking"]["count_evidence"]
    assert store.snapshot_token("u") == before
    t = turn(5, "给我推荐手机")
    default = process(h, t, value(t, "recommend"))
    assert len(default["ranking"]["items"]) == 3 and "requested_count" not in default["ranking"]
    assert h._last_request["k"] == 3


def test_current_number_can_exceed_fallback_without_pretending_missing_products_exist(tmp_path):
    h = host(tmp_path)
    t = turn(1, "预算三千，推荐五款手机")
    result = process(h, t, payload(t, recommend=True))
    assert ids(result) == ["B", "C"] and result["ranking"]["requested_count"] == 5
    assert h._last_request["k"] == 5
    assert not JsonlMemoryStore(tmp_path / "memory").list_memories("u", as_of=after(t))


def test_ambiguous_count_cannot_partially_commit_brand_or_budget(tmp_path):
    h = host(tmp_path)
    t = turn(1, "我长期喜欢品牌C。预算三千，推荐两三款手机")
    raw = payload(t, recommend=True)
    raw["attribute_request"]["conditions"][0]["evidence"] = [quote(t, "预算三千")]
    brand = value(t, entity="品牌C", relation="prefer")
    brand["candidate"]["assertions"][0]["evidence"] = [quote(t, "我长期喜欢品牌C")]
    raw["intents"] = ["recommend", "express_preference"]
    raw["candidate"]["assertions"] = brand["candidate"]["assertions"]
    client = FixedStages(raw, raw)
    result = h.process_user(t, complete=client, as_of=after(t), strategy="popularity", k=3)
    assert result["response_action"] == "clarify" and result["ranking"] is None
    assert "新长期偏好尚未保存" in result["clarification"]
    assert result["new_memory_ids"] == [] and len(client.seen) == 2
    assert h._attribute_request is None and h._last_request is None
    assert JsonlMemoryStore(tmp_path / "memory").list_memories("u", as_of=after(t)) == ()


def test_long_term_evidence_must_not_store_temporary_quantity_clause(tmp_path):
    t = turn(1, "我长期喜欢品牌C，给我推荐两款手机")
    raw = value(t, "recommend", entity="品牌C", relation="prefer")
    raw["intents"] = ["express_preference", "recommend"]
    raw["candidate"]["request"]["requested_entity"] = None
    with pytest.raises(ExtractionError, match="mixed_retention_evidence"):
        decode_interaction(json.dumps(raw, ensure_ascii=False), turns=(t,))
    raw["candidate"]["assertions"][0]["evidence"] = [quote(t, "我长期喜欢品牌C")]
    h = session(tmp_path)
    result = process(h, t, raw)
    assert result["response_action"] == "recommend" and len(result["ranking"]["items"]) == 2
    record = JsonlMemoryStore(tmp_path / "memory").list_memories("u", as_of=after(t))[0]
    assert record.evidence_text == "我长期喜欢品牌C" and "两款" not in record.evidence_text


def test_consultation_does_not_become_a_quantity_request(tmp_path):
    h = session(tmp_path)
    t = turn(1, "你怎么看两款品牌C手机")
    result = process(h, t, value(t, entity="品牌C"))
    assert result["response_action"] == "answer" and result["ranking"] is None
    assert h._last_request is None


@pytest.mark.parametrize(
    "text", ["给我推荐手机，不要只推荐三款", "不推荐3款，给我推荐手机", "不止推荐三款"]
)
def test_negated_quantity_without_replacement_cannot_fall_back_to_forbidden_default(tmp_path, text):
    h = session(tmp_path)
    t = turn(1, text)
    result = process(h, t, value(t, "recommend"))
    assert result["response_action"] == "clarify" and result["ranking"] is None
    assert h._last_request is None


def test_repeated_same_count_cannot_expand_evidence_response_without_bound(tmp_path):
    h = session(tmp_path)
    t = turn(1, "给我推荐一款手机；" * 100)
    result = process(h, t, value(t, "recommend"))
    ranking = result["ranking"]
    assert ranking["requested_count"] == 1 and len(ranking["items"]) == 1
    assert len(json.dumps(ranking["count_evidence"]).encode()) < 2048
    assert ranking["count_evidence"][0]["turn_id"] == t.turn_id


def test_omitted_recommend_flag_cannot_bypass_temporary_quantity_retention_guard(tmp_path):
    h = session(tmp_path)
    t = turn(1, "我长期喜欢品牌C，给我推荐两款手机")
    raw = value(t, entity="品牌C", relation="prefer")
    with pytest.raises(AnalystError):
        process(h, t, raw)
    assert JsonlMemoryStore(tmp_path / "memory").list_memories("u", as_of=after(t)) == ()


@pytest.mark.parametrize(
    "text", ["我看过不少手机请推荐两款手机", "我想好了请推荐两款手机", "不过请推荐两款手机"]
)
def test_past_words_outside_current_request_cannot_erase_explicit_count(tmp_path, text):
    h = session(tmp_path)
    t = turn(1, text)
    result = process(h, t, value(t, "recommend"))
    assert result["ranking"]["requested_count"] == 2
    assert len(result["ranking"]["items"]) == 2
