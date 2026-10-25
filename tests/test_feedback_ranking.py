"""可信反馈接真实全池排名，不把固定模型工程用例当用户收益。"""

from datetime import timedelta

import pytest
from test_memory import T0
from test_memory_products import AT, engine, request

from memrecagent.memory_feedback import FeedbackTarget, HostFeedbackSession
from memrecagent.memory_feedback_store import JsonFeedbackStore
from memrecagent.memory_use_policy import JsonMemoryUseStore, MemoryUsePolicy


def block(
    root, *, kind="product", target="A", category="phone", user="demo-user", at=T0, event="block-a"
):
    store = JsonFeedbackStore(root / "feedback")
    host = HostFeedbackSession(store, user_id=user, session_id="S1")
    display = host.register_display(FeedbackTarget(kind, category, target), displayed_at=at)
    host.submit(display, event_id=event, at=at)
    return store, host


def ids(result):
    return [item["item_id"] for item in result["items"]]


def test_product_block_filters_before_topk_and_undo_restores_after_reopen(tmp_path):
    store, host = block(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = engine(tmp_path).recommend(**(request() | {"k": 1}))
    assert ids(result) == ["B"]
    assert result["filter_counts"] == {"feedback_product": 1}
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    host.undo("block-a", action_id="undo-a", at=T0 + timedelta(hours=1))
    assert ids(engine(tmp_path).recommend(**request())) == ["A", "B", "C"]


def test_brand_request_overrides_brand_only_not_exact_product_block(tmp_path):
    block(tmp_path, kind="brand", target="品牌A")
    assert ids(engine(tmp_path).recommend(**request())) == ["B", "C"]
    assert ids(engine(tmp_path).recommend(**request(requested_entity="品牌A"))) == ["A"]
    block(tmp_path, event="product-block")
    result = engine(tmp_path).recommend(**request(requested_entity="品牌A"))
    assert ids(result) == []
    assert result["filter_counts"]["feedback_product"] == 1
    assert ids(engine(tmp_path).recommend(**request())) == ["B", "C"]


@pytest.mark.parametrize("mode", ["other", "unknown", "off", "denied", "other-user"])
def test_outside_owner_or_permitted_memory_does_not_apply_personal_feedback(
    tmp_path, monkeypatch, mode
):
    block(tmp_path)
    if mode == "denied":
        JsonMemoryUseStore(tmp_path / "use-policy").set(
            "demo-user", MemoryUsePolicy(personalisation=False)
        )
    if mode != "other-user":

        def forbidden(*args, **kwargs):
            pytest.fail("不应读取本人有效反馈视图")

        monkeypatch.setattr(JsonFeedbackStore, "active_targets", forbidden, raising=False)
    changes = dict(use_long_term_memory=mode != "off")
    if mode in ("other", "unknown"):
        changes["beneficiary_id"] = "uncle" if mode == "other" else None
    if mode == "other-user":
        changes["user_id"] = "someone-else"
    assert ids(engine(tmp_path).recommend(**(request() | changes))) == ["A", "B", "C"]


def test_feedback_scope_category_and_time_are_exact(tmp_path):
    block(tmp_path, kind="brand", target="品牌B", category="laptop", event="other-category")
    block(tmp_path, target="品牌C", event="product-is-not-brand")
    block(tmp_path, at=AT, event="future")
    ranker = engine(tmp_path)
    assert ids(ranker.recommend(**request())) == ["A", "B", "C"]
    assert ids(ranker.recommend(**(request() | {"as_of": AT + timedelta(seconds=1)}))) == ["B", "C"]


def test_feedback_invalidates_tokens_even_when_undo_restores_same_ranking(tmp_path):
    ranker = engine(tmp_path)
    old = ranker.state_token("demo-user")
    store, host = block(tmp_path)
    blocked = ranker.state_token("demo-user")
    assert blocked != old
    host.undo("block-a", action_id="undo-a", at=T0 + timedelta(hours=1))
    undone = ranker.state_token("demo-user")
    assert undone not in (old, blocked)
    store.delete("demo-user", FeedbackTarget("product", "phone", "A"), AT)
    assert ranker.state_token("demo-user") not in (old, blocked, undone)
    assert ids(ranker.recommend(**request())) == ["A", "B", "C"]


def test_feedback_four_groups_read_same_source_with_only_memory_switch(tmp_path):
    from test_memory_ablation import fitted_ranker_inputs

    from memrecagent.memory_ablation import run_ranking_ablation
    from memrecagent.memory_store import JsonlMemoryStore

    service, _, facts = fitted_ranker_inputs(tmp_path / "fixture")
    store = JsonlMemoryStore(tmp_path / "clean")
    block(store.root, target="A", user="u")
    result = run_ranking_ablation(
        service, store, facts, request=request() | dict(user_id="u", strategy="bpr")
    )
    assert {g: ids(r["ranking"]) for g, r in result.items()} == {
        "G00": ["A", "B", "C"],
        "G01": ["A", "C", "B"],
        "G10": ["B", "C"],
        "G11": ["C", "B"],
    }
    assert result["G11"]["algorithm_effective"] is True


def test_mid_rank_feedback_change_aborts_publication(tmp_path, monkeypatch):
    import memrecagent.memory_products as module

    ranker = engine(tmp_path)
    original = module.constraint_failure
    changed = False

    def mutate(*args, **kwargs):
        nonlocal changed
        if not changed:
            changed = True
            block(tmp_path)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "constraint_failure", mutate)
    with pytest.raises(ValueError, match="request_state_changed"):
        ranker.recommend(**request())


def test_product_block_does_not_hide_sibling_product_of_same_brand(tmp_path):
    from dataclasses import replace

    block(tmp_path)
    ranker = engine(tmp_path)
    ranker._facts = tuple(
        replace(f, entity="共享品牌") if f.item_id in ("A", "B") else f for f in ranker._facts
    )
    assert ids(ranker.recommend(**request())) == ["B", "C"]


def test_actual_conversation_reads_feedback_then_invalidates_cached_rpc_on_undo(tmp_path):
    from test_conversation_bridge import request as rpc_request
    from test_conversation_products import session
    from test_memory_analyst_host import FixedStages
    from test_memory_interaction import after, turn, value

    from memrecagent.conversation_bridge import ConversationBridge

    h = session(tmp_path)
    t = turn(3, "给我推荐手机")
    _, host = block(tmp_path / "memory", user="u", at=after(turn(1, "前一轮")))
    raw = value(t, "recommend")
    client = FixedStages(raw, raw)
    bridge = ConversationBridge(h, complete=client)
    payload = rpc_request(t, id="recommend")
    first = bridge.handle(payload)
    assert first["ok"] and ids(first["result"]["ranking"]) == ["B", "C"]
    host.undo("block-a", action_id="undo-a", at=after(t) + timedelta(seconds=1))
    replay = bridge.handle(payload)
    assert not replay["ok"] and replay["error"]["code"] == "conversation_state"
    assert len(client.seen) == 2
    newer = turn(5, "现在再推荐手机")
    next_raw = value(newer, "recommend")
    client.responses += (next_raw, next_raw)
    following = bridge.handle(rpc_request(newer, id="next"))
    assert following["ok"] and ids(following["result"]["ranking"]) == ["A", "B", "C"]
