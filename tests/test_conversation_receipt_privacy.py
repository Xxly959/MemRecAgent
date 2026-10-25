"""删除/撤权后清本实例旧回执原文，同时保留不重放与真实成本边界。"""

import json
from datetime import timedelta

import pytest
from test_conversation_bridge import request
from test_conversation_products import session
from test_memory_analyst_host import FixedStages
from test_memory_interaction import after, turn, value

from memrecagent.conversation_bridge import ConversationBridge
from memrecagent.deepseek_client import ProviderError
from memrecagent.memory_store import JsonlMemoryStore
from memrecagent.memory_use_policy import JsonMemoryUseStore, MemoryUsePolicy


def setup(tmp_path):
    h = session(tmp_path)
    t1 = turn(1, "我长期喜欢品牌C")
    t2 = turn(3, "给我推荐手机")
    raw1 = value(t1, entity="品牌C", relation="prefer")
    raw2 = value(t2, "recommend")
    client = FixedStages(raw1, raw1, raw2, raw2)
    bridge = ConversationBridge(h, complete=client)
    p1 = request(t1, id="save")
    p2 = request(t2, id="rank")
    first = bridge.handle(p1)
    second = bridge.handle(p2)
    assert first["ok"] and second["ok"]
    return h, bridge, client, p1, p2, second, t2


def remove(tmp_path, t):
    JsonlMemoryStore(tmp_path / "memory").delete_entity(
        "u", category="phone", entity="品牌C", deleted_at=after(t) + timedelta(seconds=1)
    )


def assert_expired(result):
    assert not result["ok"] and result["error"]["code"] == "conversation_state"
    assert result["error"]["memory_commit_status"] == "unknown"
    assert "new_memory_ids" not in result["error"]


def test_request_fingerprints_do_not_keep_user_or_assistant_text(tmp_path):
    _, b, _, p1, _, _, _ = setup(tmp_path)
    assert all("我长期喜欢品牌C" not in entry[0] for entry in b._receipts.values())
    t = turn(4, "这是只应留在会话的助手原文", role="assistant")
    assert b.handle(request(t, id="assistant", method="assistant_turn"))["ok"]
    assert all(t.content not in entry[0] for entry in b._receipts.values())
    assert b.handle(p1)["ok"]  # 同状态仍能准确重放


@pytest.mark.parametrize("change", ["delete", "deny", "aba"])
def test_stale_receipt_drops_raw_data_preserves_cost_and_never_recharges(tmp_path, change):
    h, b, client, p1, p2, second, t = setup(tmp_path)
    if change == "delete":
        remove(tmp_path, t)
    else:
        policy = JsonMemoryUseStore(tmp_path / "memory/use-policy")
        policy.set("u", MemoryUsePolicy(personalisation=False))
        if change == "aba":
            policy.set("u", MemoryUsePolicy())
    replay = b.handle(p2)
    assert_expired(replay)
    assert len(replay["error"]["analysis_calls"]) == 2
    assert replay["error"]["analysis_calls"][0]["status"] == "ok"
    assert len(client.seen) == 4 and h._memory.turns == ()
    assert h._last_request is None and h._attribute_request is None and b._comparison is None
    assert "我长期喜欢品牌C" not in str(b._receipts)
    assert "综合" not in str(b._receipts)
    assert all("result" not in json.loads(entry[1]) for entry in b._receipts.values())
    for _ in range(3):
        assert_expired(b.handle(p1))
        assert_expired(b.handle(p2))
    assert len(client.seen) == 4
    assert second["ok"]  # 不原地篡改caller已经持有的旧对象


def test_same_id_changed_input_stays_conflict_after_invalidation(tmp_path):
    _, b, client, p1, _, _, t = setup(tmp_path)
    remove(tmp_path, t)
    assert_expired(b.handle(p1))
    changed = request(turn(1, "另一条全新的用户表达"), id="save")
    assert b.handle(changed)["error"]["code"] == "id_conflict"
    assert len(client.seen) == 4
    assert (
        JsonlMemoryStore(tmp_path / "memory").active_memories(
            "u", as_of=after(t) + timedelta(seconds=3)
        )
        == ()
    )


def test_normal_own_commit_expires_older_receipt_but_keeps_fresh_history_and_current_result(
    tmp_path,
):
    h, b, client, p1, _, _, _ = setup(tmp_path)
    t = turn(5, "以后长期排除品牌A，也请推荐手机")
    raw = value(t, "recommend", entity="品牌A", relation="exclude")
    raw["intents"] = ["recommend", "express_preference"]
    raw["candidate"]["request"]["requested_entity"] = None
    client.responses += (raw, raw)
    payload = request(t, id="new")
    result = b.handle(payload)
    assert result["ok"] and len(result["result"]["new_memory_ids"]) == 1
    assert [i["item_id"] for i in result["result"]["ranking"]["items"]] == ["C", "B"]
    assert h._memory.turns and any(t.content == row.content for row in h._memory.turns)
    assert b.handle(payload) == result and len(client.seen) == 6
    assert_expired(b.handle(p1))


def test_cached_provider_failure_keeps_original_attempt_metadata_after_delete(tmp_path):
    h = session(tmp_path)
    client = FixedStages(ProviderError("transport"))
    b = ConversationBridge(h, complete=client)
    t = turn(1, "给我推荐手机")
    payload = request(t)
    assert b.handle(payload)["error"]["code"] == "analysis_failed"
    remove(tmp_path, t)
    result = b.handle(payload)
    assert_expired(result)
    assert len(result["error"]["analysis_calls"]) == 1
    assert result["error"]["analysis_calls"][0]["status"] == "provider_error"
    assert len(client.seen) == 1


def test_state_check_cannot_replay_cached_valid_true_after_deletion(tmp_path):
    _, b, client, _, _, second, t = setup(tmp_path)
    payload = dict(
        version=1,
        id="check",
        method="check_state",
        params=dict(state_token=second["result"]["state_token"]),
    )
    assert b.handle(payload)["ok"]
    remove(tmp_path, t)
    assert_expired(b.handle(payload))
    assert len(client.seen) == 4


def test_root_binding_change_and_read_failure_purge_context_without_new_analysis(tmp_path):
    h, b, client, _, p2, _, _ = setup(tmp_path)
    old_root = h._memory._store.root
    h._memory._store.root = tmp_path / "unbound"
    assert_expired(b.handle(p2))
    assert h._memory.turns == () and h._memory._host is None
    h._memory._store.root = old_root
    path = JsonlMemoryStore(old_root)._path("u")
    path.write_text('{"private-marker": invalid')
    assert_expired(b.handle(p2))
    assert "我长期喜欢品牌C" not in str(b._receipts) and "private-marker" not in str(b._receipts)
    assert h._memory._host is None and len(client.seen) == 4


def test_deletion_after_commit_before_bridge_receipt_is_not_reported_as_zero_writes(
    tmp_path, monkeypatch
):
    h = session(tmp_path)
    t = turn(1, "我长期喜欢品牌C")
    raw = value(t, entity="品牌C", relation="prefer")
    client = FixedStages(raw, raw)
    original = h.process_user

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        remove(tmp_path, t)
        return result

    monkeypatch.setattr(h, "process_user", changed)
    b = ConversationBridge(h, complete=client)
    payload = request(t)
    result = b.handle(payload)
    assert_expired(result)
    assert len(result["error"]["analysis_calls"]) == 2 and len(client.seen) == 2
    assert h._memory.turns == () and "我长期喜欢品牌C" not in str(b._receipts)
    assert b.handle(payload) == result and len(client.seen) == 2


def test_invalidation_keeps_capacity_tombstones_instead_of_freeing_ids_for_replay(tmp_path):
    h = session(tmp_path)
    b = ConversationBridge(h, complete=FixedStages())
    token = h.state_token()
    for i in range(256):
        result = b.handle(
            dict(version=1, id=f"check{i}", method="check_state", params=dict(state_token=token))
        )
        assert result["ok"]
    t = turn(1, "占位")
    remove(tmp_path, t)
    assert_expired(
        b.handle(dict(version=1, id="check0", method="check_state", params=dict(state_token=token)))
    )
    assert (
        b.handle(dict(version=1, id="new", method="check_state", params=dict(state_token=token)))[
            "error"
        ]["code"]
        == "session_full"
    )


def test_delete_while_decoding_cached_reply_does_not_publish_stale_body(tmp_path, monkeypatch):
    import memrecagent.conversation_bridge as module

    _, bridge, client, _, query, _, t = setup(tmp_path)
    original = module.json.loads
    changed = False

    def loads(raw, *args, **kwargs):
        nonlocal changed
        value = original(raw, *args, **kwargs)
        if not changed and isinstance(raw, str) and '"conversation_context"' in raw:
            changed = True
            remove(tmp_path, t)
        return value

    monkeypatch.setattr(module.json, "loads", loads)
    assert_expired(bridge.handle(query))
    assert changed and len(client.seen) == 4
    assert "我长期喜欢品牌C" not in str(bridge._receipts)
