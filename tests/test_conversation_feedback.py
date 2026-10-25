"""实际会话展示绑定与可信用户反馈，不让模型造点击。"""

from datetime import timedelta

import pytest
from test_conversation_products import process, session
from test_memory_extraction import subject
from test_memory_interaction import after, turn, value
from test_memory_interaction_extraction import quote

from memrecagent.memory_commit import CommitError
from memrecagent.memory_feedback_store import JsonFeedbackStore
from memrecagent.memory_use_policy import JsonMemoryUseStore, MemoryUsePolicy


def ids(ranking):
    return [p["item_id"] for p in ranking["items"]]


def shown(tmp_path, *, other=False, requested=False):
    h = session(tmp_path)
    t = turn(
        1, "我叔叔要手机给他推荐" if other else "这次给我看品牌A" if requested else "给我推荐手机"
    )
    raw = value(t, "recommend", **({"entity": "品牌A"} if requested else {}))
    if other:
        raw["candidate"]["request"]["beneficiary"] = subject("other", [quote(t, "我叔叔")])
    data = process(h, t, raw)
    h.observe_assistant(turn(2, "实际展示了排序商品", "assistant"))
    exposed = h.register_feedback_display(
        h.recommendation_token(), displayed_at=after(turn(2, "x"))
    )
    return h, data, exposed


def submit(h, display, *, scope="persistent", kind="product", index=1, event="event-a"):
    return h.submit_feedback(
        display["display_token"],
        index=index,
        kind=kind,
        scope=scope,
        event_id=event,
        at=after(turn(2, "x")) + timedelta(seconds=1),
    )


def test_actual_display_block_and_undo_rerank_without_model_and_persist(tmp_path):
    h, original, display = shown(tmp_path)
    assert display["n_items"] == 3
    blocked = submit(h, display)
    assert ids(blocked["ranking"]) == ["B", "C"]
    assert blocked["state_token"] != original["state_token"]
    assert blocked["event_id"] == "event-a" and blocked["scope"] == "persistent"
    with pytest.raises(CommitError):
        submit(h, display, event="second")
    fresh = session(tmp_path, "fresh")
    t = turn(5, "给我推荐手机")
    assert ids(process(fresh, t, value(t, "recommend"))["ranking"]) == ["B", "C"]
    undone = h.undo_feedback("event-a", action_id="undo-a", at=after(turn(4, "x")))
    assert ids(undone["ranking"]) == ["A", "B", "C"]
    assert undone["state_token"] not in (original["state_token"], blocked["state_token"])


def test_registration_requires_observed_display_and_current_nonce(tmp_path):
    h = session(tmp_path)
    t = turn(1, "给我推荐手机")
    process(h, t, value(t, "recommend"))
    nonce = h.recommendation_token()
    with pytest.raises(CommitError):
        h.register_feedback_display(nonce, displayed_at=after(t))
    h.observe_assistant(turn(2, "已显示", "assistant"))
    with pytest.raises(CommitError):
        h.register_feedback_display("0" * 32, displayed_at=after(turn(2, "x")))
    assert h.register_feedback_display(nonce, displayed_at=after(turn(2, "x")))["n_items"] == 3


@pytest.mark.parametrize(
    "index,kind,scope",
    [
        (0, "product", "persistent"),
        (4, "product", "persistent"),
        (True, "product", "persistent"),
        (1, "category", "persistent"),
        (1, "product", "anything"),
    ],
)
def test_invalid_user_selection_never_writes(tmp_path, index, kind, scope):
    h, _, display = shown(tmp_path)
    before = h.state_token()
    with pytest.raises((ValueError, CommitError)):
        submit(h, display, index=index, kind=kind, scope=scope)
    assert h.state_token() == before and not (tmp_path / "memory/feedback").exists()


def test_current_request_feedback_is_shared_by_four_groups_then_expires(tmp_path):
    h, original, display = shown(tmp_path)
    blocked = submit(h, display, scope="request")
    assert ids(blocked["ranking"]) == ["B", "C"]
    for memory in (False, True):
        assert ids(h.rank_current(use_long_term_memory=memory)) == ["B", "C"]
    assert not (tmp_path / "memory/feedback").exists()
    t = turn(5, "给我重新推荐手机")
    following = process(h, t, value(t, "recommend"))
    assert ids(following["ranking"]) == ["A", "B", "C"]
    assert following["state_token"] != original["state_token"]


def test_feedback_for_other_person_cannot_become_owners_persistent_preference(tmp_path):
    h, _, display = shown(tmp_path, other=True)
    with pytest.raises(CommitError):
        submit(h, display)
    blocked = submit(h, display, scope="request")
    assert ids(blocked["ranking"]) == ["B", "C"]
    assert not (tmp_path / "memory/feedback").exists()
    fresh = session(tmp_path, "new")
    t = turn(5, "给我推荐手机")
    assert ids(process(fresh, t, value(t, "recommend"))["ranking"]) == ["A", "B", "C"]


def test_latest_brand_block_click_wins_over_older_current_brand_request(tmp_path):
    h, _, display = shown(tmp_path, requested=True)
    assert ids(submit(h, display, kind="brand")["ranking"]) == []
    fresh = session(tmp_path, "new")
    t = turn(5, "这次给我看品牌A")
    assert ids(process(fresh, t, value(t, "recommend", entity="品牌A"))["ranking"]) == ["A"]


def test_policy_change_rejects_stale_display_before_any_feedback_write(tmp_path):
    h, _, display = shown(tmp_path)
    JsonMemoryUseStore(tmp_path / "memory/use-policy").set(
        "u", MemoryUsePolicy(long_term_updates=False)
    )
    with pytest.raises(CommitError):
        submit(h, display)
    assert not (tmp_path / "memory/feedback").exists()


def rpc(bridge, method, params, identity):
    return bridge.handle(dict(version=1, id=identity, method=method, params=params))


def test_rpc_display_block_replay_and_undo_no_additional_analysis(tmp_path):
    from test_conversation_bridge import request
    from test_memory_analyst_host import FixedStages

    from memrecagent.conversation_bridge import ConversationBridge

    h = session(tmp_path)
    t = turn(1, "给我推荐手机")
    raw = value(t, "recommend")
    client = FixedStages(raw, raw)
    b = ConversationBridge(h, complete=client)
    q = request(t, id="query")
    assert b.handle(q)["ok"]
    assert b.handle(request(turn(2, "实际展示", "assistant"), id="shown", method="assistant_turn"))[
        "ok"
    ]
    d = rpc(
        b,
        "feedback_display",
        dict(request_token=h.recommendation_token(), displayed_at=after(turn(2, "x")).isoformat()),
        "display",
    )
    assert d["ok"]
    params = dict(
        display_token=d["result"]["display_token"],
        index=1,
        kind="product",
        scope="request",
        event_id="rpc-event",
        at=after(turn(3, "x")).isoformat(),
    )
    blocked = rpc(b, "feedback_submit", params, "block")
    assert blocked["ok"] and ids(blocked["result"]["ranking"]) == ["B", "C"]
    assert rpc(b, "feedback_submit", params, "block") == blocked
    assert not b.handle(q)["ok"]  # 相同磁盘但临时反馈也使旧缓存失效。
    undone = rpc(
        b,
        "feedback_undo",
        dict(event_id="rpc-event", action_id="rpc-undo", at=after(turn(4, "x")).isoformat()),
        "undo",
    )
    assert undone["ok"] and ids(undone["result"]["ranking"]) == ["A", "B", "C"]
    assert len(client.seen) == 2


def test_rpc_arbitrary_target_is_rejected(tmp_path):
    from test_memory_analyst_host import FixedStages

    from memrecagent.conversation_bridge import ConversationBridge

    h, _, display = shown(tmp_path)
    b = ConversationBridge(h, complete=FixedStages())
    params = dict(
        display_token=display["display_token"],
        index=1,
        kind="product",
        scope="persistent",
        event_id="bad",
        at=after(turn(3, "x")).isoformat(),
        target_id="B",
    )
    result = rpc(b, "feedback_submit", params, "arbitrary")
    assert not result["ok"] and result["error"]["code"] == "invalid_request"
    assert not (tmp_path / "memory/feedback").exists()


def test_deleted_feedback_source_releases_new_display_and_ranking_references(tmp_path):
    from memrecagent.memory_feedback import FeedbackTarget

    h, _, display = shown(tmp_path)
    submit(h, display)
    h.observe_assistant(turn(4, "展示B、C", "assistant"))
    h.register_feedback_display(h.recommendation_token(), displayed_at=after(turn(4, "x")))
    JsonFeedbackStore(tmp_path / "memory/feedback").delete(
        "u", FeedbackTarget("product", "phone", "A"), after(turn(5, "x"))
    )
    h.refresh_read_state()
    assert h._feedback_display is None and h._last_ranking is None
    assert h._feedback_overlay == {} and h._feedback._displays == {}
    assert h._last_request is None


def test_known_feedback_write_with_failed_rerank_is_not_reported_as_no_write(tmp_path, monkeypatch):
    from test_memory_analyst_host import FixedStages

    from memrecagent.conversation_bridge import ConversationBridge
    from memrecagent.memory_feedback import FeedbackTarget

    h, _, display = shown(tmp_path)
    bridge = ConversationBridge(h, complete=FixedStages())

    def fail(**query):
        raise ValueError("rank unavailable")

    monkeypatch.setattr(h._ranker, "recommend", fail)
    params = dict(
        display_token=display["display_token"],
        index=1,
        kind="product",
        scope="persistent",
        event_id="committed",
        at=after(turn(3, "x")).isoformat(),
    )
    result = rpc(bridge, "feedback_submit", params, "failed-rerank")
    assert not result["ok"] and result["error"]["feedback_commit_status"] == "unknown"
    assert JsonFeedbackStore(tmp_path / "memory/feedback").blocked(
        "u", FeedbackTarget("product", "phone", "A"), after(turn(4, "x"))
    )
    with pytest.raises(CommitError):
        h.recommendation_token()
    assert rpc(bridge, "feedback_submit", params, "failed-rerank")["ok"] is False
