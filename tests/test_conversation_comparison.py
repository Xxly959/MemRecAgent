"""同一次实际用户输入的只读排名RPC，不能补gold或重复记忆分析。"""

import pytest
from test_conversation_bridge import request
from test_memory_analyst_host import FixedStages
from test_memory_extraction import subject
from test_memory_interaction import after, turn, value
from test_memory_interaction_extraction import quote
from test_request_attributes import host, payload

from memrecagent.conversation_bridge import ConversationBridge
from memrecagent.memory_store import JsonlMemoryStore


def rpc(bridge, method, params=None, identity="rpc"):
    return bridge.handle(dict(version=1, id=identity, method=method, params=params or {}))


def prefix(tmp_path):
    h = host(tmp_path)
    first = turn(1, "我长期喜欢品牌C。")
    latest = turn(3, "这次预算三千，给我推荐手机。")
    prefer = value(first, entity="品牌C", relation="prefer")
    budget = payload(latest, recommend=True)
    client = FixedStages(prefer, prefer, budget, budget)
    bridge = ConversationBridge(h, complete=client)
    assert bridge.handle(request(first, id="pref"))["ok"]
    displayed = turn(2, "已记录这条偏好。", "assistant")
    assert bridge.handle(request(displayed, id="display", method="assistant_turn"))["ok"]
    result = bridge.handle(request(latest, id="query"))["result"]
    return h, bridge, client, latest, result


def branch_params(binding, *, memory=True, strategy="popularity"):
    return dict(
        request_token=binding["request_token"], use_long_term_memory=memory, strategy=strategy
    )


def test_readonly_four_variants_keep_actual_budget_history_and_never_reanalyse(tmp_path):
    h, bridge, client, latest, original = prefix(tmp_path)
    before = JsonlMemoryStore(tmp_path / "memory").snapshot_token("u")
    binding = rpc(bridge, "comparison_binding")["result"]
    assert binding["prompt"] == latest.content
    assert binding["state_token"] == original["state_token"]
    for index, (memory, strategy) in enumerate(
        (
            (False, "popularity"),
            (True, "popularity"),
            (False, "bpr_cold_popularity"),
            (True, "bpr_cold_popularity"),
        )
    ):
        params = branch_params(binding, memory=memory, strategy=strategy)
        result = rpc(bridge, "rank_current", params, identity=f"arm-{index}")
        assert result["ok"]
        data = result["result"]
        assert data["analysis_calls"] == data["new_memory_ids"] == []
        assert data["conversation_context"] == original["conversation_context"]
        assert data["history_truncated"] == original["history_truncated"]
        assert data["ranking"]["use_long_term_memory"] is memory
        assert data["ranking"]["requested_strategy"] == strategy
        assert "A" not in [p["item_id"] for p in data["ranking"]["items"]]
        if strategy == "popularity":
            assert [p["item_id"] for p in data["ranking"]["items"]] == (
                ["C", "B"] if memory else ["B", "C"]
            )
        assert rpc(bridge, "rank_current", params, identity=f"arm-{index}") == result
    assert len(client.seen) == 4
    assert JsonlMemoryStore(tmp_path / "memory").snapshot_token("u") == before
    assert rpc(
        bridge,
        "check_comparison",
        {k: binding[k] for k in ("request_token", "state_token")},
        identity="check",
    )["result"] == {"valid": True}


def test_same_memory_snapshot_new_question_invalidates_old_binding_and_cached_reply(tmp_path):
    h, bridge, client, _, original = prefix(tmp_path)
    binding = rpc(bridge, "comparison_binding")["result"]
    params = branch_params(binding)
    assert rpc(bridge, "rank_current", params, identity="old-rank")["ok"]
    t = turn(5, "再给我推荐手机。")
    raw = value(t, "recommend")
    client.responses += (raw, raw)
    newer = bridge.handle(request(t, id="new-query"))
    assert newer["ok"] and newer["result"]["state_token"] == original["state_token"]
    assert (
        rpc(bridge, "rank_current", params, identity="old-rank")["error"]["code"]
        == "conversation_state"
    )
    assert (
        rpc(
            bridge,
            "check_comparison",
            {k: binding[k] for k in ("request_token", "state_token")},
            identity="old-check",
        )["ok"]
        is False
    )
    new_binding = rpc(bridge, "comparison_binding", identity="new-binding")["result"]
    assert new_binding["request_token"] != binding["request_token"]


def test_deleted_state_blocks_even_cached_readonly_receipt(tmp_path):
    h, bridge, client, t, _ = prefix(tmp_path)
    binding = rpc(bridge, "comparison_binding")["result"]
    params = branch_params(binding)
    assert rpc(bridge, "rank_current", params, identity="cached")["ok"]
    JsonlMemoryStore(tmp_path / "memory").delete_entity(
        "u", category="phone", entity="品牌C", deleted_at=after(t)
    )
    assert (
        rpc(bridge, "rank_current", params, identity="cached")["error"]["code"]
        == "conversation_state"
    )
    assert len(client.seen) == 4


@pytest.mark.parametrize("kind", ["identity", "budget", "bool", "token"])
def test_invalid_readonly_params_cannot_override_bound_request(tmp_path, kind):
    _, bridge, client, _, _ = prefix(tmp_path)
    binding = rpc(bridge, "comparison_binding")["result"]
    params = branch_params(binding)
    if kind == "identity":
        params["beneficiary_id"] = "other"
    elif kind == "budget":
        params["constraints"] = []
    elif kind == "bool":
        params["use_long_term_memory"] = 0
    else:
        params["request_token"] = "forged"
    assert rpc(bridge, "rank_current", params, identity="bad")["error"]["code"] == "invalid_request"
    assert len(client.seen) == 4


def test_nonrecommendation_clears_comparison_even_without_memory_change(tmp_path):
    _, bridge, client, _, _ = prefix(tmp_path)
    t = turn(5, "晚上好。")
    raw = value(t, "unrelated")
    client.responses += (raw, raw)
    assert bridge.handle(request(t, id="greeting"))["ok"]
    assert rpc(bridge, "comparison_binding")["error"]["code"] == "conversation_state"


def test_no_request_yet_cannot_prepare_comparison_or_call_model(tmp_path):
    client = FixedStages()
    bridge = ConversationBridge(host(tmp_path), complete=client)
    response = rpc(bridge, "comparison_binding")
    assert response["error"]["code"] == "conversation_state"
    assert client.seen == []


def test_other_recipient_keeps_own_current_budget_but_uses_no_owner_profile(tmp_path):
    _, bridge, client, _, _ = prefix(tmp_path)
    t = turn(5, "这次我叔叔预算四千五，给他推荐手机。")
    who = subject("other", [quote(t, "我叔叔")])
    raw = payload(t, amount=4500, who=who, recommend=True)
    client.responses += (raw, raw)
    assert bridge.handle(request(t, id="other-query"))["ok"]
    binding = rpc(bridge, "comparison_binding")["result"]
    response = rpc(
        bridge,
        "rank_current",
        branch_params(binding, strategy="bpr_cold_popularity"),
        identity="other-rank",
    )
    assert response["ok"]
    ranking = response["result"]["ranking"]
    assert [i["item_id"] for i in ranking["items"]] == ["A", "B", "C"]
    assert ranking["used_strategy"] == "popularity"
    assert all(i["preference_signal"] == 0 for i in ranking["items"])
    assert len(client.seen) == 6


def test_change_during_read_cannot_publish_result_with_freshly_relabelled_state(
    tmp_path, monkeypatch
):
    h, bridge, client, t, _ = prefix(tmp_path)
    binding = rpc(bridge, "comparison_binding")["result"]
    original = h._ranker.recommend

    def changed(**kwargs):
        result = original(**kwargs)
        JsonlMemoryStore(tmp_path / "memory").delete_entity(
            "u", category="phone", entity="品牌C", deleted_at=after(t)
        )
        return result

    monkeypatch.setattr(h._ranker, "recommend", changed)
    response = rpc(bridge, "rank_current", branch_params(binding), identity="changed-rank")
    assert response["error"]["code"] == "conversation_state"
    assert len(client.seen) == 4
