"""软属性影响完整排序但不筛掉商品，且本人/当前/四组边界独立。"""

from dataclasses import replace
from datetime import timedelta

import pytest
from test_attribute_memory_store import attribute
from test_memory import T0
from test_product_attributes import constraint, request, value, with_values

from memrecagent.memory_store import JsonlMemoryStore
from memrecagent.memory_use_policy import JsonMemoryUseStore, MemoryUsePolicy


def ids(result):
    return [i["item_id"] for i in result["items"]]


def priced(root):
    return with_values(root, {"A": [value(value=4000)], "B": [value(value=2500)], "C": []})


def test_saved_soft_preference_actually_changes_rank_with_no_filter(tmp_path):
    JsonlMemoryStore(tmp_path).add(attribute(is_hard=False))
    result = priced(tmp_path).recommend(**request())
    assert ids(result) == ["B", "A", "C"]
    assert result["n_filtered"] == 0
    assert [i["score"] for i in result["items"]] == [1.25, 0.25, 0]
    assert result["items"][0]["attribute_preference_signal"] == 1
    assert result["items"][2]["attribute_signals"][0]["status"] == "unknown"
    assert result["soft_attribute_preferences"][0]["scope"] == "long_term"


@pytest.mark.parametrize("mode", ["off", "other", "denied", "soft-off"])
def test_long_term_soft_signal_respects_memory_identity_and_control(tmp_path, mode):
    JsonlMemoryStore(tmp_path).add(attribute(is_hard=False))
    if mode == "denied":
        JsonMemoryUseStore(tmp_path / "use-policy").set(
            "demo-user", MemoryUsePolicy(personalisation=False)
        )
    changes = {}
    if mode == "off":
        changes["use_long_term_memory"] = False
    if mode == "other":
        changes["beneficiary_id"] = "uncle"
    if mode == "soft-off":
        changes["use_soft_attributes"] = False
    assert ids(priced(tmp_path).recommend(**(request() | changes))) == ["A", "B", "C"]


def test_current_soft_is_shared_when_memory_disabled_and_does_not_remove_hard_budget(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    store.add(attribute(value=2000))
    result = priced(tmp_path).recommend(**request(), soft_preferences=(constraint(value=4500),))
    assert result["items"] == []  # 软要求不能覆盖硬预算/未知属性硬过滤。
    current = priced(tmp_path).recommend(
        **request(), soft_preferences=(constraint(),), use_long_term_memory=False
    )
    assert ids(current) == ["B", "A", "C"]
    assert current["soft_attribute_preferences"][0]["scope"] == "current"


def test_current_predicate_overrides_same_attribute_soft_without_double_count(tmp_path):
    JsonlMemoryStore(tmp_path).add(attribute(is_hard=False, value=2000))
    result = priced(tmp_path).recommend(**request(), soft_preferences=(constraint(value=4500),))
    assert ids(result) == ["A", "B", "C"]
    assert len(result["soft_attribute_preferences"]) == 1
    assert result["items"][0]["attribute_preference_signal"] == 1
    assert result["overridden_soft_attribute_memory_ids"] == ["a1"]


def test_multiple_predicates_of_same_attribute_conjoin_and_never_count_twice(tmp_path):
    result = priced(tmp_path).recommend(
        **request(),
        soft_preferences=(constraint(value=3000), replace(constraint(), operator=">=", value=2000)),
    )
    assert ids(result) == ["B", "A", "C"]
    assert result["items"][0]["attribute_preference_signal"] == 1
    assert result["items"][1]["attribute_preference_signal"] == -1


@pytest.mark.parametrize(
    "changes",
    [
        dict(user_id="wrong"),
        dict(beneficiary_id="uncle"),
        dict(request_id="stale"),
        dict(available_at=T0 + timedelta(days=2)),
    ],
)
def test_soft_request_binding_cannot_cross_person_request_or_time(tmp_path, changes):
    with pytest.raises(ValueError):
        priced(tmp_path).recommend(
            **request(), soft_preferences=(replace(constraint(), **changes),)
        )


def test_future_fact_soft_is_unknown_not_satisfied_or_filtered(tmp_path):
    ranker = priced(tmp_path)
    ranker._facts = tuple(
        replace(f, attributes=(value(value=1000, at=T0 + timedelta(days=2)),))
        for f in ranker._facts
    )
    result = ranker.recommend(**request(), soft_preferences=(constraint(),))
    assert ids(result) == ["A", "B", "C"]
    assert all(i["attribute_preference_signal"] == 0 for i in result["items"])
    assert result["n_filtered"] == 0


def test_real_tool_decodes_soft_current_and_honours_soft_ablation(tmp_path):
    import json

    from test_memory_product_tool import fixture, run_tool

    fixture(tmp_path)
    query_path = tmp_path / "request.json"
    query = json.loads(query_path.read_text()) | dict(
        request_id="R1",
        soft_preferences=[
            dict(
                user_id="demo-user",
                session_id="S2",
                beneficiary_id="self",
                attribute="price_cny",
                operator="<=",
                value=3000,
                unit="CNY",
                evidence_ref="S2:T1",
                available_at=T0.isoformat(),
                request_id="R1",
            )
        ],
    )
    query_path.write_text(json.dumps(query))
    facts_path = tmp_path / "facts.json"
    facts = json.loads(facts_path.read_text())
    for fact in facts["facts"]:
        if fact["item_id"] in ("A", "B"):
            fact["attributes"] = [
                dict(
                    attribute="price_cny",
                    value=4000 if fact["item_id"] == "A" else 2500,
                    unit="CNY",
                    evidence_ref="catalog:fixed",
                    available_at=T0.isoformat(),
                )
            ]
    facts_path.write_text(json.dumps(facts))
    result = run_tool(tmp_path)
    assert result.returncode == 0, result.stderr
    assert ids(json.loads(result.stdout)) == ["B", "A", "C"]
    query_path.write_text(json.dumps(query | dict(use_soft_attributes=False)))
    assert ids(json.loads(run_tool(tmp_path).stdout)) == ["A", "B", "C"]


def test_standalone_delivery_receipt_expires_when_feedback_changes(tmp_path):
    import json

    from test_feedback_ranking import block
    from test_memory_product_tool import fixture, run_tool

    fixture(tmp_path)
    result = json.loads(run_tool(tmp_path).stdout)
    block(tmp_path / "memory")
    assert run_tool(tmp_path, "--check-token", result["delivery_token"]).returncode == 2


def test_actual_soft_language_candidate_saves_ranks_and_reuses_current_once(tmp_path):
    from test_attribute_language import durable
    from test_conversation_products import process
    from test_memory_interaction import turn
    from test_request_attributes import host, plain

    h = host(tmp_path)
    t = turn(1, "以后价格更倾向三千以内，不是硬性预算，请长期保存并推荐手机。")
    raw = durable(t, recommend=True)
    raw["attribute_changes"][0]["is_hard"] = False
    result = process(h, t, raw)
    assert ids(result["ranking"]) == ["B", "C", "A"]
    assert len(result["ranking"]["soft_attribute_preferences"]) == 1
    assert result["ranking"]["soft_attribute_preferences"][0]["scope"] == "current"
    for enabled in (False, True):
        assert ids(h.rank_current(use_long_term_memory=enabled)) == ["B", "C", "A"]
    t2 = turn(3, "给我推荐手机")
    assert ids(process(host(tmp_path, "new"), t2, plain(t2))["ranking"]) == ["B", "C", "A"]


def test_mean_is_over_attributes_including_unknown_not_duplicated_predicates(tmp_path):
    ranker = with_values(
        tmp_path,
        {
            "A": [value(value=4000), value("battery_life_hours", 16, "h")],
            "B": [value(value=2500), value("battery_life_hours", 8, "h")],
            "C": [value("battery_life_hours", 24, "h")],
        },
    )
    result = ranker.recommend(
        **request(),
        soft_preferences=(constraint(), constraint("battery_life_hours", ">=", 12, "h")),
    )
    by_id = {i["item_id"]: i for i in result["items"]}
    assert by_id["A"]["attribute_preference_signal"] == 0
    assert by_id["B"]["attribute_preference_signal"] == 0
    assert by_id["C"]["attribute_preference_signal"] == 0.5
    assert by_id["C"]["score"] == 0.375


@pytest.mark.parametrize("flag", [1, None, "false"])
def test_soft_flag_cannot_be_truthy_conversion(tmp_path, flag):
    with pytest.raises(ValueError):
        priced(tmp_path).recommend(**request(), use_soft_attributes=flag)


def test_other_soft_language_state_is_ram_only_and_not_inherited_by_owner(tmp_path):
    from test_attribute_language import durable
    from test_conversation_products import process
    from test_memory_extraction import subject
    from test_memory_interaction import after, turn
    from test_memory_interaction_extraction import quote
    from test_request_attributes import host, plain

    h = host(tmp_path)
    t = turn(1, "我叔叔手机价格更倾向三千以内，不是硬预算，给他推荐。")
    raw = durable(t, who=subject("other", [quote(t, "我叔叔")]), recommend=True)
    raw["attribute_changes"][0]["is_hard"] = False
    result = process(h, t, raw)
    assert ids(result["ranking"]) == ["B", "C", "A"]
    assert JsonlMemoryStore(tmp_path / "memory").list_attribute_memories("u", as_of=after(t)) == ()
    newer = turn(3, "给我推荐手机")
    assert ids(process(host(tmp_path, "new"), newer, plain(newer))["ranking"]) == ["A", "B", "C"]


@pytest.mark.parametrize("soft", [False, True])
def test_deleted_attribute_old_request_cannot_reactivate_but_new_evidence_can(tmp_path, soft):
    store = JsonlMemoryStore(tmp_path)
    store.delete_attribute(
        "demo-user", category="phone", attribute="price_cny", deleted_at=T0 + timedelta(hours=1)
    )
    field = "soft_preferences" if soft else "constraints"
    with pytest.raises(ValueError, match="deleted_attribute_evidence"):
        priced(tmp_path).recommend(**request(), **{field: (constraint(),)})
    fresh = replace(constraint(), available_at=T0 + timedelta(hours=2))
    result = priced(tmp_path).recommend(**request(), **{field: (fresh,)})
    assert ids(result) == (["B", "A", "C"] if soft else ["B"])
