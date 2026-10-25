"""可信卡片反馈的真实文件与会话边界，不使用语言模型。"""

from datetime import UTC, datetime, timedelta

import pytest

from memrecagent.memory_feedback import FeedbackTarget, HostFeedbackSession
from memrecagent.memory_feedback_store import JsonFeedbackStore

BASE = datetime(2026, 10, 22, tzinfo=UTC)
PRODUCT = FeedbackTarget("product", "phone", "synthetic-phone-1")
OTHER = FeedbackTarget("product", "phone", "synthetic-phone-2")
BRAND = FeedbackTarget("brand", "phone", "小米")


def at(n):
    return BASE + timedelta(seconds=n)


def host(tmp_path, user="u", session="s"):
    return HostFeedbackSession(JsonFeedbackStore(tmp_path), user_id=user, session_id=session)


def click(h, target=PRODUCT, event="e1", n=1, scope="persistent"):
    card = h.register_display(target, displayed_at=at(0))
    h.submit(card, event_id=event, at=at(n), scope=scope)
    return card


def test_exact_target_persists_without_brand_expansion(tmp_path):
    h = host(tmp_path)
    click(h)
    restored = host(tmp_path, session="new")
    assert restored.blocked(PRODUCT, as_of=at(2))
    assert not restored.blocked(OTHER, as_of=at(2))
    assert not restored.blocked(BRAND, as_of=at(2))
    assert not restored.blocked(FeedbackTarget("product", "camera", PRODUCT.target_id), as_of=at(2))
    assert not host(tmp_path, user="other").blocked(PRODUCT, as_of=at(2))


def test_brand_block_does_not_guess_products(tmp_path):
    h = host(tmp_path)
    click(h, BRAND)
    assert h.blocked(BRAND, as_of=at(2))
    assert not h.blocked(PRODUCT, as_of=at(2))


def test_explicit_request_temporarily_overrides_without_erasing(tmp_path):
    h = host(tmp_path)
    click(h)
    assert not h.blocked(PRODUCT, as_of=at(2), explicit_request=True)
    assert h.blocked(PRODUCT, as_of=at(2))
    assert host(tmp_path, session="new").blocked(PRODUCT, as_of=at(2))


def test_session_only_has_no_disk_effect_and_disappears_on_restart(tmp_path):
    h = host(tmp_path)
    click(h, scope="request")
    assert h.blocked(PRODUCT, as_of=at(2))
    assert not list(tmp_path.iterdir())
    assert not host(tmp_path).blocked(PRODUCT, as_of=at(2))


@pytest.mark.parametrize("scope", ["persistent", "request"])
def test_idempotency_undo_and_old_replay(tmp_path, scope):
    h = host(tmp_path)
    card = click(h, scope=scope)
    h.submit(card, event_id="e1", at=at(1), scope=scope)
    h.undo("e1", action_id="undo1", at=at(2))
    h.undo("e1", action_id="undo1", at=at(2))
    assert not h.blocked(PRODUCT, as_of=at(3))
    h.submit(card, event_id="e1", at=at(1), scope=scope)
    assert not h.blocked(PRODUCT, as_of=at(3))


@pytest.mark.parametrize("scope", ["persistent", "request"])
def test_old_undo_does_not_cancel_new_block_or_resurrect_old(tmp_path, scope):
    h = host(tmp_path)
    card = click(h, scope=scope)
    h.submit(card, event_id="e2", at=at(2), scope=scope)
    h.undo("e1", action_id="undo1", at=at(3))
    assert h.blocked(PRODUCT, as_of=at(4))
    h.undo("e2", action_id="undo2", at=at(4))
    assert not h.blocked(PRODUCT, as_of=at(5))


def test_persistent_undo_survives_restart(tmp_path):
    h = host(tmp_path)
    click(h)
    new = host(tmp_path, session="new")
    new.undo("e1", action_id="undo1", at=at(2))
    assert not host(tmp_path).blocked(PRODUCT, as_of=at(3))


@pytest.mark.parametrize("change", ["target", "time", "scope"])
def test_event_payload_conflict_refused(tmp_path, change):
    h = host(tmp_path)
    card = click(h)
    if change == "target":
        card = h.register_display(OTHER, displayed_at=at(0))
    with pytest.raises(ValueError):
        h.submit(
            card,
            event_id="e1",
            at=at(2 if change == "time" else 1),
            scope="request" if change == "scope" else "persistent",
        )
    assert h.blocked(PRODUCT, as_of=at(3))
    assert not h.blocked(OTHER, as_of=at(3))


@pytest.mark.parametrize("mode", ["fake", "other_session", "other_user", "future"])
def test_only_actual_host_display_for_same_actor(tmp_path, mode):
    h = host(tmp_path)
    card = h.register_display(PRODUCT, displayed_at=at(10 if mode == "future" else 0))
    actor = (
        host(tmp_path, session="other")
        if mode == "other_session"
        else (host(tmp_path, user="other") if mode == "other_user" else h)
    )
    with pytest.raises(ValueError):
        actor.submit("fake" if mode == "fake" else card, event_id="e1", at=at(1))
    assert not list(tmp_path.iterdir())


def test_time_queries_and_nonboolean_override(tmp_path):
    h = host(tmp_path)
    click(h)
    assert not h.blocked(PRODUCT, as_of=at(1))
    assert h.blocked(PRODUCT, as_of=at(2))
    with pytest.raises(ValueError):
        h.blocked(PRODUCT, as_of=at(2), explicit_request="false")
    with pytest.raises(ValueError):
        h.register_display(PRODUCT, displayed_at=datetime(2026, 10, 22))


@pytest.mark.parametrize("scope", ["persistent", "request"])
def test_delete_removes_target_and_blocks_old_display_replay(tmp_path, scope):
    h = host(tmp_path)
    card = click(h, scope=scope)
    assert h.delete(PRODUCT, at=at(3)) >= 0
    assert not h.blocked(PRODUCT, as_of=at(4))
    with pytest.raises(ValueError):
        h.submit(card, event_id="new-old-card", at=at(4), scope=scope)
    with pytest.raises(ValueError):
        h.submit(card, event_id="e1", at=at(1), scope=scope)
    with pytest.raises(ValueError):
        h.undo("e1", action_id="late-undo", at=at(4))
    fresh = h.register_display(PRODUCT, displayed_at=at(4))
    h.submit(fresh, event_id="fresh", at=at(5), scope=scope)
    assert h.blocked(PRODUCT, as_of=at(6))


def test_external_delete_invalidates_existing_session_cards_and_blocks(tmp_path):
    h = host(tmp_path)
    card = click(h, scope="request")
    host(tmp_path, session="new").delete(PRODUCT, at=at(3))
    assert not h.blocked(PRODUCT, as_of=at(4))
    with pytest.raises(ValueError):
        h.submit(card, event_id="e2", at=at(4), scope="request")


def test_delete_physically_erases_target_but_keeps_other_and_replay_receipts(tmp_path):
    h = host(tmp_path)
    click(h, BRAND)
    click(h, OTHER, event="other", n=2)
    h.undo("e1", action_id="undo", at=at(3))
    h.delete(BRAND, at=at(4))
    content = next(tmp_path.glob("*.json")).read_text()
    assert "小米" not in content
    assert OTHER.target_id in content
    assert host(tmp_path).blocked(OTHER, as_of=at(5))
    fresh = h.register_display(BRAND, displayed_at=at(5))
    with pytest.raises(ValueError):
        h.submit(fresh, event_id="e1", at=at(6))


def test_backdated_delete_or_new_event_refused(tmp_path):
    h = host(tmp_path)
    card = click(h, n=5)
    with pytest.raises(ValueError):
        h.delete(PRODUCT, at=at(4))
    with pytest.raises(ValueError):
        h.submit(card, event_id="old", at=at(3))
    assert h.blocked(PRODUCT, as_of=at(6))


@pytest.mark.parametrize("bad", ["truncated", "extra", "duplicate", "wrong_owner"])
def test_corrupt_file_refused_without_overwrite(tmp_path, bad):
    import json

    h = host(tmp_path)
    click(h)
    path = next(tmp_path.glob("*.json"))
    data = json.loads(path.read_text())
    if bad == "extra":
        data["surprise"] = True
    if bad == "wrong_owner":
        data["owner_digest"] = "0" * 64
    raw = "{" if bad == "truncated" else json.dumps(data)
    if bad == "duplicate":
        raw = raw[:-1] + ',"schema_version":1}'
    path.write_text(raw)
    with pytest.raises(ValueError):
        h.blocked(PRODUCT, as_of=at(2))
    assert path.read_text() == raw


def test_atomic_failure_preserves_previous_file(tmp_path, monkeypatch):
    import memrecagent.memory_feedback_store as module

    h = host(tmp_path)
    card = click(h)
    path = next(tmp_path.glob("*.json"))
    old = path.read_bytes()

    def broken(*args):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(module.os, "replace", broken)
    with pytest.raises(OSError):
        h.submit(card, event_id="e2", at=at(2))
    assert path.read_bytes() == old
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "kind,category,identifier",
    [("all", "phone", "x"), ("brand", "", "x"), ("product", "phone", ""), ("brand", "phone", 1)],
)
def test_target_requires_explicit_precise_identity(kind, category, identifier):
    with pytest.raises(ValueError):
        FeedbackTarget(kind, category, identifier)


def test_next_request_clears_transient_block_and_old_cards(tmp_path):
    h = host(tmp_path)
    card = click(h, scope="request")
    click(h, OTHER, event="persist", n=2)
    h.start_request()
    assert not h.blocked(PRODUCT, as_of=at(3))
    assert h.blocked(OTHER, as_of=at(3))
    with pytest.raises(ValueError):
        h.submit(card, event_id="old-request", at=at(4), scope="request")
    assert len(h._events) == 0


@pytest.mark.parametrize("bad", ["extra", "missing", "wrong_scope", "bad_session", "bad_ref"])
def test_store_rejects_invalid_candidate_before_publish(tmp_path, bad):
    import copy
    import json

    h = host(tmp_path)
    click(h)
    path = next(tmp_path.glob("*.json"))
    old = path.read_bytes()
    event = copy.deepcopy(json.loads(old)["events"][0])
    event["event"] = "a" * 64
    event["at"] = at(2).isoformat()
    if bad == "extra":
        event["surprise"] = True
    elif bad == "missing":
        del event["session"]
    elif bad == "wrong_scope":
        event["scope"] = "request"
    elif bad == "bad_session":
        event["session"] = True
    else:
        event.update(action="undo", ref="b" * 64, display=None, displayed_at=None)
    with pytest.raises(ValueError):
        h.store.append("u", event)
    assert path.read_bytes() == old


def test_external_delete_prunes_local_derived_state_on_next_access(tmp_path):
    h = host(tmp_path)
    click(h, BRAND, scope="request")
    host(tmp_path, session="new").delete(BRAND, at=at(3))
    assert not h.blocked(BRAND, as_of=at(4))
    assert not h._events
    assert not h._displays
