"""语言属性候选必须经真实双阶段与宿主，不能按gold补配置。"""

import json
from dataclasses import replace

import pytest
from test_conversation_products import process, session
from test_memory import T0
from test_memory_analyst_host import FixedStages
from test_memory_extraction import subject
from test_memory_interaction import after, confirmation, turn, value
from test_memory_interaction_extraction import quote

from memrecagent.memory_analyst import AnalystError
from memrecagent.memory_commit import CommitError
from memrecagent.memory_extraction import ExtractionError
from memrecagent.memory_interaction_extraction import decode_interaction
from memrecagent.memory_store import JsonlMemoryStore
from memrecagent.product_attributes import ProductAttributeValue
from memrecagent.request_attributes import bind_constraints, numbers_in_text


def payload(t, *, op="start", amount=3000, recommend=False, who=None, remove=()):
    raw = value(t, "recommend" if recommend else "express_preference")
    raw["schema_version"] = 3
    raw["attribute_request"] = dict(
        operation=op,
        scope="request",
        subject=who or subject(),
        conditions=[]
        if amount is None
        else [
            dict(
                attribute="price_cny", operator="<=", value=amount, unit="CNY", evidence=[quote(t)]
            )
        ],
        remove_attributes=list(remove),
        evidence=[quote(t)],
    )
    if recommend and who:
        raw["candidate"]["request"]["beneficiary"] = who
    return raw


def plain(t, intent="recommend", **extra):
    return value(t, intent, **extra) | dict(schema_version=3, attribute_request=None)


def host(tmp_path, sid="s"):
    result = session(tmp_path, sid)
    result._ranker._facts = tuple(
        replace(
            f,
            attributes=(
                ProductAttributeValue(
                    "price_cny",
                    {"A": 4000, "B": 2500, "C": 2000}[f.item_id],
                    "CNY",
                    "synthetic:price",
                    T0,
                ),
            ),
        )
        for f in result._ranker._facts
    )
    return result


def ids(result):
    return [r["item_id"] for r in result["ranking"]["items"]]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("预算三千", 3000),
        ("两千五以内", 2500),
        ("三千五百", 3500),
        ("一万二", 12000),
        ("三千零五", 3005),
        ("6.6英寸", 6.6),
        ("3k以内", 3000),
    ],
)
def test_numeric_grounding_supports_literal_chinese_and_explicit_suffixes(text, expected):
    assert expected in numbers_in_text(text)


def test_v3_budget_is_readonly_candidate_and_v2_remains_strict():
    t = turn(1, "这次预算三千，给我推荐手机。")
    raw = payload(t, recommend=True)
    candidate = decode_interaction(json.dumps(raw), turns=(t,))
    assert candidate.attribute_request.conditions[0].value == 3000
    assert candidate.attribute_request.conditions[0].evidence[0].turn_id == t.turn_id
    raw["schema_version"] = 2
    with pytest.raises(ExtractionError):
        decode_interaction(json.dumps(raw), turns=(t,))


@pytest.mark.parametrize(
    "mutation", ["invent_value", "assistant", "old_evidence", "extra_id", "long_term"]
)
def test_untrusted_attribute_proposal_cannot_bypass_evidence_or_host_binding(mutation):
    old, t = turn(1, "预算四千"), turn(3, "这次预算三千")
    assistant = turn(2, "你的预算四千。", "assistant")
    raw = payload(t)
    event = raw["attribute_request"]
    if mutation == "invent_value":
        event["conditions"][0]["value"] = 4000
    if mutation == "assistant":
        event["conditions"][0]["evidence"] = [quote(assistant)]
    if mutation == "old_evidence":
        event["conditions"][0]["evidence"] = [quote(old)]
    if mutation == "extra_id":
        event["request_id"] = "model-chosen"
    if mutation == "long_term":
        event["scope"] = "long_term"
    expected = {
        "invent_value": "attribute_value_evidence",
        "assistant": "quote_role",
        "old_evidence": "attribute_evidence",
        "extra_id": "fields",
        "long_term": "attribute_scope",
    }
    with pytest.raises(ExtractionError) as caught:
        decode_interaction(json.dumps(raw), turns=(old, assistant, t))
    assert caught.value.code == expected[mutation]


def test_actual_multiturn_budget_update_end_and_restart_never_write_long_term(tmp_path):
    h = host(tmp_path)
    t = turn(1, "这次预算三千。")
    assert process(h, t, payload(t))["new_memory_ids"] == []
    t = turn(3, "给我推荐手机。")
    first = process(h, t, plain(t))
    assert ids(first) == ["B", "C"]
    t = turn(5, "本次预算放宽到四千五。")
    process(h, t, payload(t, op="update", amount=4500))
    t = turn(7, "按本次要求再挑一轮手机。")
    assert ids(process(h, t, plain(t))) == ["A", "B", "C"]
    t = turn(9, "先不买了，结束这次选购。")
    process(h, t, payload(t, op="end", amount=None))
    t = turn(11, "重新给我推荐手机。")
    assert ids(process(h, t, plain(t))) == ["A", "B", "C"]
    assert JsonlMemoryStore(tmp_path / "memory").list_memories("u", as_of=after(t)) == ()
    fresh = host(tmp_path, "fresh")
    t = turn(13, "给我推荐手机。")
    assert ids(process(fresh, t, plain(t))) == ["A", "B", "C"]


def test_switching_beneficiary_does_not_apply_owner_budget(tmp_path):
    h = host(tmp_path)
    t = turn(1, "这次我预算三千。")
    process(h, t, payload(t))
    current = turn(2, "给我先推荐手机。")
    assert ids(process(h, current, plain(current))) == ["B", "C"]
    t = turn(3, "这次给我叔叔推荐手机。")
    raw = plain(t)
    raw["candidate"]["request"]["beneficiary"] = subject("other", [quote(t, "我叔叔")])
    assert ids(process(h, t, raw)) == ["A", "B", "C"]


def test_bad_final_budget_does_not_partially_commit_valid_brand_preference(tmp_path):
    h = host(tmp_path)
    t = turn(1, "我长期喜欢品牌C，这次预算三千。")
    raw = payload(t, amount=4000)
    raw["candidate"]["assertions"] = value(t, entity="品牌C", relation="prefer")["candidate"][
        "assertions"
    ]
    with pytest.raises(AnalystError):
        process(h, t, raw)
    assert JsonlMemoryStore(tmp_path / "memory").list_memories("u", as_of=after(t)) == ()


def test_failed_brand_commit_does_not_publish_new_ram_budget(tmp_path, monkeypatch):
    h = host(tmp_path)
    t = turn(1, "这次预算三千。")
    process(h, t, payload(t))
    original = h._memory.commit

    def fail(*args, **kwargs):
        raise CommitError("stale")

    monkeypatch.setattr(h._memory, "commit", fail)
    t = turn(3, "这次预算放宽到四千五。")
    with pytest.raises(CommitError):
        process(h, t, payload(t, op="update", amount=4500))
    monkeypatch.setattr(h._memory, "commit", original)
    t = turn(5, "给我推荐手机。")
    assert ids(process(h, t, plain(t))) == ["B", "C"]


def test_evidence_reset_after_store_deletion_invalidates_old_ram_budget(tmp_path):
    h = host(tmp_path)
    t = turn(1, "这次预算三千。")
    process(h, t, payload(t))
    JsonlMemoryStore(tmp_path / "memory").delete_entity(
        "u", category="phone", entity="品牌A", deleted_at=after(t)
    )
    t = turn(3, "给我推荐手机。")
    with pytest.raises(CommitError) as caught:
        process(h, t, plain(t))
    assert caught.value.code == "stale"
    t = turn(5, "删除旧上下文后，重新给我推荐手机。")
    assert ids(process(h, t, plain(t))) == ["A", "B", "C"]


def test_attribute_only_other_person_is_registered_for_next_actual_analysis(tmp_path):
    h = host(tmp_path)
    t = turn(1, "我叔叔这次预算三千。")
    who = subject("other", [quote(t, "我叔叔")])
    process(h, t, payload(t, who=who))
    t = turn(3, "给他推荐手机。")
    raw = plain(t)
    raw["candidate"]["request"]["beneficiary"] = who
    client = FixedStages(raw, raw)
    result = h.process_user(t, complete=client, as_of=after(t), strategy="popularity", k=3)
    assert ids(result) == ["B", "C"]
    assert client.seen[0]["output_contract"]["known_other_subjects"] == [who]
    assert JsonlMemoryStore(tmp_path / "memory").list_memories("u", as_of=after(t)) == ()


def test_explicit_remove_and_start_without_conditions_clear_previous_budget(tmp_path):
    h = host(tmp_path)
    t = turn(1, "这次预算三千。")
    process(h, t, payload(t))
    t = turn(3, "本次取消预算上限。")
    process(h, t, payload(t, op="update", amount=None, remove=("price_cny",)))
    t = turn(5, "给我推荐手机。")
    assert ids(process(h, t, plain(t))) == ["A", "B", "C"]
    t = turn(7, "新的选购任务预算三千。")
    process(h, t, payload(t))
    t = turn(9, "再开始一项不限预算的新任务，推荐手机。")
    assert ids(process(h, t, payload(t, amount=None, recommend=True))) == ["A", "B", "C"]


def test_unknown_owner_cannot_be_silently_assigned_or_combined_with_recommendation():
    t = turn(1, "他预算三千，给我推荐手机。")
    raw = payload(t, who=subject("unknown"), recommend=True)
    with pytest.raises(ExtractionError) as caught:
        decode_interaction(json.dumps(raw), turns=(t,))
    assert caught.value.code == "attribute_subject"


def test_explicit_end_is_lifecycle_not_new_preference_and_may_be_unrelated(tmp_path):
    h = host(tmp_path)
    t = turn(1, "这次预算三千。")
    process(h, t, payload(t))
    t = turn(3, "这次选购结束，先不买了。")
    raw = payload(t, op="end", amount=None)
    raw["intents"] = ["unrelated"]
    assert process(h, t, raw)["new_memory_ids"] == []
    t = turn(5, "开始新任务，给我推荐手机。")
    assert ids(process(h, t, plain(t))) == ["A", "B", "C"]


def test_unrelated_still_cannot_start_or_update_attribute_constraints():
    t = turn(1, "这次预算三千。")
    raw = payload(t)
    raw["intents"] = ["unrelated"]
    with pytest.raises(ExtractionError) as caught:
        decode_interaction(json.dumps(raw), turns=(t,))
    assert caught.value.code == "attribute_mixed"


@pytest.mark.parametrize("confirmed", ["self", "other"])
def test_actual_subject_clarification_resumes_original_budget_not_an_empty_request(
    tmp_path, confirmed
):
    h = host(tmp_path)
    t = turn(1, "给他按三千预算推荐手机。")
    raw = payload(t, recommend=True, who=subject("unknown"))
    raw["candidate"]["unresolved"] = [dict(field="subject", evidence=[quote(t)])]
    result = process(h, t, raw)
    assert result["response_action"] == "clarify"
    q = turn(2, result["clarification"], "assistant")
    h.observe_assistant(q)
    t = turn(
        3, "是我本人，只对本次选购有效。" if confirmed == "self" else "是我叔叔，只对本次选购有效。"
    )
    raw = confirmation(t, q)
    raw["confirmation"]["scope"] = "current"
    if confirmed == "other":
        raw["confirmation"].update(subject="other", person_reference=quote(t, "我叔叔"))
    raw.update(schema_version=3, attribute_request=None)
    resumed = process(h, t, raw)
    assert ids(resumed) == ["B", "C"]
    assert resumed["new_memory_ids"] == []
    constraints = bind_constraints(h._attribute_request, user_id="u", session_id="s")
    assert constraints[0].available_at == t.available_at
    assert "T1" in constraints[0].evidence_ref and "T3" in constraints[0].evidence_ref


def test_unclear_reply_keeps_budget_pending_until_real_confirmation(tmp_path):
    h = host(tmp_path)
    t = turn(1, "给他按三千预算推荐手机。")
    raw = payload(t, recommend=True, who=subject("unknown"))
    raw["candidate"]["unresolved"] = [dict(field="subject", evidence=[quote(t)])]
    pending = process(h, t, raw)
    q = turn(2, pending["clarification"], "assistant")
    h.observe_assistant(q)
    t = turn(3, "我还没确定。")
    raw = confirmation(t, q, outcome="unclear") | dict(schema_version=3, attribute_request=None)
    pending = process(h, t, raw)
    assert pending["response_action"] == "clarify"
    q = turn(4, pending["clarification"], "assistant")
    h.observe_assistant(q)
    t = turn(5, "是我本人，只对这次选购有效。")
    raw = confirmation(t, q) | dict(schema_version=3, attribute_request=None)
    raw["confirmation"]["scope"] = "current"
    assert ids(process(h, t, raw)) == ["B", "C"]
