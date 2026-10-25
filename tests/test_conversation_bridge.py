"""持续Python会话承接真实输入，模型不能伪造宿主参数。"""

from test_conversation_products import session
from test_memory_analyst_host import FixedStages
from test_memory_interaction import after, turn, value

from memrecagent.conversation_bridge import ConversationBridge


def request(t, *, id="r1", method="user_turn"):
    params = {
        "turn_id": t.turn_id,
        "content": t.content,
        "occurred_at": t.occurred_at.isoformat(),
        "available_at": t.available_at.isoformat(),
    }
    if method == "user_turn":
        params |= {"as_of": after(t).isoformat(), "strategy": "popularity", "k": 3}
    return {"version": 1, "id": id, "method": method, "params": params}


def test_replay_is_exact_and_does_not_recharge_or_write_again(tmp_path):
    host = session(tmp_path)
    t = turn(1, "我长期喜欢品牌C")
    raw = value(t, entity="品牌C", relation="prefer")
    client = FixedStages(raw, raw)
    bridge = ConversationBridge(host, complete=client)
    first = bridge.handle(request(t))
    assert first["ok"] and len(first["result"]["new_memory_ids"]) == 1
    assert bridge.handle(request(t)) == first and len(client.seen) == 2
    changed = request(t)
    changed["params"]["content"] = "换一个用户原文"
    assert bridge.handle(changed)["error"]["code"] == "id_conflict"
    assert len(client.seen) == 2


def test_failure_is_cached_too_and_never_falls_back_to_direct(tmp_path):
    from memrecagent.deepseek_client import ProviderError

    client = FixedStages(ProviderError("transport"))
    bridge = ConversationBridge(session(tmp_path), complete=client)
    payload = request(turn(1, "给我推荐手机"))
    result = bridge.handle(payload)
    assert result["ok"] is False and result["error"]["code"] == "analysis_failed"
    assert len(result["error"]["analysis_calls"]) == 1
    assert result["error"]["analysis_calls"][0]["status"] == "provider_error"
    assert result["error"]["new_memory_ids"] == []
    assert bridge.handle(payload) == result and len(client.seen) == 1


def test_host_records_actual_assistant_and_checks_delivery_snapshot(tmp_path):
    t = turn(1, "给我推荐手机")
    raw = value(t, "recommend")
    bridge = ConversationBridge(session(tmp_path), complete=FixedStages(raw, raw))
    response = bridge.handle(request(t))["result"]
    assert [p["item_id"] for p in response["ranking"]["items"]] == ["A", "B", "C"]
    token = response["state_token"]
    checked = bridge.handle(
        {"version": 1, "id": "check", "method": "check_state", "params": {"state_token": token}}
    )
    assert checked["ok"] and checked["result"] == {"valid": True}
    assistant = turn(2, "这是实际已输出的推荐。", "assistant")
    assert bridge.handle(request(assistant, id="assistant", method="assistant_turn"))["ok"]


def test_unknown_or_overridden_identity_never_calls_model(tmp_path):
    client = FixedStages()
    bridge = ConversationBridge(session(tmp_path), complete=client)
    payload = request(turn(1, "给我推荐手机"))
    payload["params"]["user_id"] = "attacker"
    assert bridge.handle(payload)["error"]["code"] == "invalid_request"
    payload["method"] = "write_memory"
    assert bridge.handle(payload)["ok"] is False and client.seen == []


def test_changed_state_after_ranking_is_not_relabelled_as_a_fresh_snapshot(tmp_path, monkeypatch):
    from memrecagent.memory_use_policy import JsonMemoryUseStore, MemoryUsePolicy

    host = session(tmp_path)
    original = host._ranker.recommend

    def changed(**kwargs):
        result = original(**kwargs)
        JsonMemoryUseStore(tmp_path / "memory" / "use-policy").set(
            "u", MemoryUsePolicy(personalisation=False)
        )
        return result

    monkeypatch.setattr(host._ranker, "recommend", changed)
    t = turn(1, "给我推荐手机")
    raw = value(t, "recommend")
    result = ConversationBridge(host, complete=FixedStages(raw, raw)).handle(request(t))
    assert result["ok"] is False and result["error"]["code"] == "conversation_state"


def test_oversize_receipt_keeps_cost_and_already_saved_ids(tmp_path, monkeypatch):
    host = session(tmp_path)
    original = host.process_user

    def oversized(*args, **kwargs):
        result = original(*args, **kwargs)
        return result | {"diagnostic_payload": "x" * 65537}

    monkeypatch.setattr(host, "process_user", oversized)
    t = turn(1, "我长期喜欢品牌C")
    raw = value(t, entity="品牌C", relation="prefer")
    client = FixedStages(raw, raw)
    bridge = ConversationBridge(host, complete=client)
    result = bridge.handle(request(t))
    assert result["error"]["code"] == "response_too_large"
    assert len(result["error"]["analysis_calls"]) == 2
    assert len(result["error"]["new_memory_ids"]) == 1
    assert bridge.handle(request(t)) == result and len(client.seen) == 2
