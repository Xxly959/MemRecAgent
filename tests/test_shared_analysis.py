"""共享一次真实草案回调的配对控制；固定回调不是模型准确率。"""

import json
from dataclasses import replace

import pytest
from test_memory_analyst import memory, reply
from test_memory_analyst_host import FixedStages
from test_memory_interaction import after, turn, value

from memrecagent.deepseek_client import ProviderError
from memrecagent.memory_analyst import AnalystError


def compare(t, client, **kwargs):
    from memrecagent.memory_analysis_control import compare_shared_draft

    return compare_shared_draft(
        (t,),
        complete=client,
        user_id="u",
        session_id="s",
        as_of=after(t),
        snapshot_token="synthetic-snapshot",
        active_memories=kwargs.pop("memories", ()),
        **kwargs,
    )


@pytest.mark.parametrize("order", [("analyst", "analyst_blind"), ("analyst_blind", "analyst")])
def test_shared_draft_has_three_physical_calls_and_only_visibility_differs(order):
    t = turn(1, "我现在取消对小米的排除。")
    draft = value(t, entity="小米", relation="prefer")
    final = value(t, entity="小米", relation="neutral")
    client = FixedStages(draft, final, final)
    result = compare(t, client, memories=(memory(),), arm_order=order)
    assert len(client.seen) == len(result.calls) == 3
    assert [c.stage for c in result.calls] == ["draft", "analyst", "analyst"]
    assert sum(dict(c.usage)["total_tokens"] for c in result.calls) == 15
    arms = {arm.mode: arm for arm in result.arms}
    assert arms["direct"].call_index == 0
    assert arms[order[0]].call_index == 1 and arms[order[1]].call_index == 2
    assert arms["direct"].analysis.candidate.assertions[0].relation == "prefer"
    assert all(arms[m].analysis.candidate.assertions[0].relation == "neutral" for m in order)
    assert not result.fatal
    messages = {
        mode: json.loads(json.dumps(client.seen[index + 1])) for index, mode in enumerate(order)
    }
    assert messages["analyst"].pop("draft") == draft
    assert messages["analyst_blind"].pop("draft") is None
    assert messages["analyst"] == messages["analyst_blind"]
    assert messages["analyst"]["memory_view"]["entries"][0]["relation"] == "exclude"


def test_bad_shared_draft_is_recorded_once_and_never_leaks_into_blind_branch():
    t = turn(1, "我一直喜欢小米手机。")
    final = value(t, entity="小米", relation="prefer")
    client = FixedStages(reply({}, content='{"PRIVATE_DRAFT_SENTINEL":true}'), final, final)
    result = compare(t, client, memories=(memory(),))
    assert len(result.calls) == 3
    assert [c.status for c in result.calls] == ["processing_error", "ok", "ok"]
    assert result.arms[0].analysis is None
    assert (
        client.seen[1]["memory_view"]
        == client.seen[2]["memory_view"]
        == {"entries": [], "truncated": False}
    )
    assert "PRIVATE_DRAFT_SENTINEL" in client.seen[1]["untrusted_draft"]["raw"]
    assert "PRIVATE_DRAFT_SENTINEL" not in json.dumps(client.seen[2])
    assert "untrusted_draft" not in client.seen[2]


@pytest.mark.parametrize("stage", [1, 2])
def test_provider_failure_stops_remaining_arms_without_retry(stage):
    t = turn(1, "我一直喜欢小米手机。")
    good = value(t, entity="小米", relation="prefer")
    responses = (ProviderError("transport"),) if stage == 1 else (good, ProviderError("transport"))
    client = FixedStages(*responses)
    result = compare(t, client)
    assert len(result.calls) == len(client.seen) == stage
    assert result.fatal
    assert result.arms[-1].status == "not_run"
    assert sum(a.status == "provider_error" for a in result.arms) == 1


def test_invalid_final_json_does_not_select_only_the_successful_arm():
    t = turn(1, "我一直喜欢小米手机。")
    good = value(t, entity="小米", relation="prefer")
    client = FixedStages(good, reply({}, content="{"), good)
    result = compare(t, client)
    assert len(result.calls) == 3 and not result.fatal
    assert [a.status for a in result.arms] == ["ok", "processing_error", "ok"]
    assert result.arms[1].analysis is None
    assert result.arms[2].analysis.candidate.assertions[0].relation == "prefer"


@pytest.mark.parametrize("stage", [1, 2])
def test_invalid_metadata_stops_instead_of_spending_remaining_pair_budget(stage):
    t = turn(1, "我一直喜欢小米手机。")
    good = value(t, entity="小米", relation="prefer")
    bad = reply(good, latency_seconds=True)
    client = FixedStages(bad) if stage == 1 else FixedStages(good, bad)
    result = compare(t, client)
    assert result.fatal and len(result.calls) == stage
    assert result.calls[-1].error_code == "reply_metadata"
    assert result.arms[-1].status == "not_run"


@pytest.mark.parametrize("kind", ["budget", "order", "order_value", "future", "owner"])
def test_invalid_control_inputs_fail_before_any_callback(kind):
    t = turn(1, "我一直喜欢小米手机。")
    options = {}
    if kind == "budget":
        options["max_calls"] = 2
    elif kind == "order":
        options["arm_order"] = ("analyst", "analyst")
    elif kind == "order_value":
        options["arm_order"] = (["forged"], "analyst")
    elif kind == "future":
        options["memories"] = (replace(memory(), available_at=after(t)),)
    else:
        options["memories"] = (memory(user_id="other"),)
    client = FixedStages()
    with pytest.raises((AnalystError, ValueError)):
        compare(t, client, **options)
    assert client.seen == []


def test_projection_keeps_typed_attributes_without_citation_spelling():
    from test_attribute_language import durable
    from test_memory_interaction_extraction import decode

    from memrecagent.memory_analysis_control import analysis_projection

    t = turn(1, "以后我的预算上限三千。")
    actual = analysis_projection(decode(durable(t), (t,)), (t,))
    assert actual == {
        "intents": ["express_preference"],
        "request": {"is_recommendation": False},
        "assertions": [],
        "clues": [],
        "unresolved": [],
        "attribute_request": None,
        "attribute_changes": [
            {
                "subject": "self",
                "person": None,
                "category": "phone",
                "attribute": "price_cny",
                "operator": "<=",
                "value": 3000,
                "unit": "CNY",
                "is_hard": True,
                "active": True,
                "scope": "long_term",
            }
        ],
        "confirmation": None,
    }


def test_projection_does_not_collapse_all_other_people_to_one_identity():
    from test_memory_extraction import subject
    from test_memory_interaction_extraction import decode, quote

    from memrecagent.memory_analysis_control import analysis_projection

    t = turn(1, "我姨妈长期喜欢三星。")
    raw = value(t, entity="三星", relation="prefer")
    raw["candidate"]["assertions"][0]["subject"] = subject("other", [quote(t, "我姨妈")])
    actual = analysis_projection(decode(raw, (t,)), (t,))
    assert actual["assertions"] == [
        {
            "subject": "other",
            "person": "我姨妈",
            "category": "phone",
            "entity": "三星",
            "relation": "prefer",
            "scope": "long_term",
        }
    ]
