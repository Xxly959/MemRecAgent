"""用途设置是用户/宿主权限，不是模型推断的用户偏好。"""

import json

import pytest
from test_memory import sample_memory
from test_memory_interaction import after, model, turn, value

from memrecagent.memory_commit import CommitError, MemoryCommitSession
from memrecagent.memory_interaction import ConversationMemorySession
from memrecagent.memory_interaction_extraction import decode_interaction
from memrecagent.memory_store import JsonlMemoryStore
from memrecagent.memory_use_policy import JsonMemoryUseStore, MemoryUsePolicy


def setup(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    store.add(sample_memory(user_id="u", entity="小米", relation="exclude"))
    return store, ConversationMemorySession(store, user_id="u", session_id="s")


def test_default_and_persisted_user_isolation(tmp_path):
    settings = JsonMemoryUseStore(tmp_path)
    assert settings.get("u") == MemoryUsePolicy()
    assert not list(tmp_path.iterdir())
    policy = MemoryUsePolicy(False, False, False)
    settings.set("u", policy)
    assert JsonMemoryUseStore(tmp_path).get("u") == policy
    assert settings.get("v") == MemoryUsePolicy()
    raw = next(tmp_path.glob("*.json")).read_text()
    assert '"u"' not in raw


@pytest.mark.parametrize("field", ["personalisation", "long_term_updates", "advertising"])
@pytest.mark.parametrize("bad", [0, 1, None, "false"])
def test_flags_strict_bool(field, bad):
    with pytest.raises(ValueError):
        MemoryUsePolicy(**{field: bad})


@pytest.mark.parametrize("mode", ["direct", "analyst"])
@pytest.mark.parametrize("intent", ["consult", "recommend"])
def test_off_never_sends_or_activates_old_personal_preference(tmp_path, mode, intent):
    store, host = setup(tmp_path)
    host.set_use_policy(MemoryUsePolicy(personalisation=False))
    assert (
        ConversationMemorySession(store, user_id="u", session_id="new")
        .get_use_policy()
        .personalisation
        is False
    )
    t = turn(1, "你怎么看小米？" if intent == "consult" else "给我推荐手机。")
    raw = value(t, intent, entity="小米" if intent == "consult" else None)
    seen = []
    preview = host.prepare_user(t, complete=model(raw, seen), as_of=after(t), analysis_mode=mode)
    assert not preview.context_memories
    assert not preview.memory_preview.proposal.decision.memories
    assert not preview.memory_preview.proposal.decision.excluded_entities
    assert all(
        not p.get("own_memories") and not p.get("memory_view", {}).get("entries") for p in seen
    )
    assert store.active_memories("u", as_of=after(t))[0].relation == "exclude"


@pytest.mark.parametrize("mode", ["direct", "analyst"])
def test_updates_off_keeps_new_expression_current_without_disk_change(tmp_path, mode):
    store, host = setup(tmp_path)
    before = store.snapshot_token("u")
    host.set_use_policy(MemoryUsePolicy(long_term_updates=False))
    t = turn(1, "以后我都更喜欢三星。")
    p = host.prepare_user(
        t,
        complete=model(value(t, entity="三星", relation="prefer")),
        as_of=after(t),
        analysis_mode=mode,
    )
    assert not p.memory_preview.proposal.proposed_memories
    host.commit(p.ticket_id, as_of=after(t))
    assert store.snapshot_token("u") == before


def test_advertising_default_deny_and_explicit_same_user_grant(tmp_path):
    store, host = setup(tmp_path)
    with pytest.raises(CommitError, match="purpose"):
        ConversationMemorySession(store, user_id="u", session_id="ad", purpose="advertising")
    host.set_use_policy(MemoryUsePolicy(advertising=True))
    ad = ConversationMemorySession(store, user_id="u", session_id="ad", purpose="advertising")
    t = turn(1, "给我推荐手机。")
    p = ad.prepare_user(t, complete=model(value(t, "recommend")), as_of=after(t))
    assert p.context_memories[0].entity == "小米"
    with pytest.raises(CommitError, match="purpose"):
        ConversationMemorySession(store, user_id="v", session_id="ad", purpose="advertising")
    host.set_use_policy(MemoryUsePolicy(personalisation=False, advertising=True))
    with pytest.raises(CommitError, match="purpose"):
        ConversationMemorySession(store, user_id="u", session_id="ad", purpose="advertising")


@pytest.mark.parametrize("mode", ["direct", "analyst"])
def test_setting_change_invalidates_old_preview_and_clears_raw_context(tmp_path, mode):
    store, host = setup(tmp_path)
    t = turn(1, "以后我都更喜欢三星。")
    p = host.prepare_user(
        t,
        complete=model(value(t, entity="三星", relation="prefer")),
        as_of=after(t),
        analysis_mode=mode,
    )
    host.set_use_policy(MemoryUsePolicy(False, False, False))
    assert not host.turns and not host.working_clues
    with pytest.raises(CommitError):
        host.commit(p.ticket_id, as_of=after(t))
    assert len(store.active_memories("u", as_of=after(t))) == 1


def test_external_setting_change_and_aba_invalidate_commit_session(tmp_path):
    store, host = setup(tmp_path)
    commit = MemoryCommitSession(store, user_id="u", session_id="structured")
    t = turn(1, "以后我都更喜欢三星。")
    candidate = decode_interaction(
        json.dumps(value(t, entity="三星", relation="prefer")), turns=(t,)
    ).candidate
    p = commit.prepare(candidate, turns=(t,), as_of=after(t))
    host.set_use_policy(MemoryUsePolicy(False, False, False))
    host.set_use_policy(MemoryUsePolicy())
    with pytest.raises(CommitError, match="stale"):
        commit.commit(p.ticket_id, as_of=after(t))
    assert len(store.active_memories("u", as_of=after(t))) == 1


def test_external_change_rejects_before_next_model_send(tmp_path):
    store, host = setup(tmp_path)
    t = turn(1, "你怎么看小米？")
    p = host.prepare_user(t, complete=model(value(t, entity="小米")), as_of=after(t))
    host.commit(p.ticket_id, as_of=after(t))
    other = ConversationMemorySession(store, user_id="u", session_id="other")
    other.set_use_policy(MemoryUsePolicy(False, False, False))
    seen = []
    latest = turn(2, "推荐手机。")
    with pytest.raises(CommitError, match="stale"):
        host.prepare_user(
            latest, complete=model(value(latest, "recommend"), seen), as_of=after(latest)
        )
    assert not seen and not host.turns


def test_disabled_updates_does_not_disable_explicit_management(tmp_path):
    store, host = setup(tmp_path)
    host.set_use_policy(MemoryUsePolicy(False, False, False))
    store.delete_entity(
        "u", category="phone", entity="小米", deleted_at=turn(1, "删除").occurred_at
    )
    assert not store.list_memories("u", as_of=after(turn(2, "查看")))


@pytest.mark.parametrize("bad", ["duplicate", "extra", "owner", "version"])
def test_corrupt_settings_fail_closed_without_rewrite(tmp_path, bad):
    settings = JsonMemoryUseStore(tmp_path)
    settings.set("u", MemoryUsePolicy())
    path = next(tmp_path.glob("*.json"))
    data = json.loads(path.read_text())
    if bad == "extra":
        data["surprise"] = True
    elif bad == "owner":
        data["owner_digest"] = "0" * 64
    elif bad == "version":
        data["schema_version"] = True
    raw = json.dumps(data)
    if bad == "duplicate":
        raw = raw[:-1] + ',"schema_version":1}'
    path.write_text(raw)
    with pytest.raises(ValueError):
        settings.get("u")
    assert path.read_text() == raw


def test_failed_settings_write_preserves_file_and_live_workspace(tmp_path, monkeypatch):
    import memrecagent.memory_use_policy as module

    store, host = setup(tmp_path)
    t = turn(1, "以后我都更喜欢三星。")
    p = host.prepare_user(
        t, complete=model(value(t, entity="三星", relation="prefer")), as_of=after(t)
    )
    host.set_use_policy(MemoryUsePolicy())
    path = next((tmp_path / "use-policy").glob("*.json"))
    before = path.read_bytes()
    p = host.prepare_user(
        t, complete=model(value(t, entity="三星", relation="prefer")), as_of=after(t)
    )

    def fail(*args):
        raise OSError("synthetic")

    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError):
        host.set_use_policy(MemoryUsePolicy(False, False, False))
    assert path.read_bytes() == before and host.turns == (t,)
    monkeypatch.undo()
    host.commit(p.ticket_id, as_of=after(t))


def test_revoke_ad_purpose_in_same_session_succeeds_but_blocks_future_use(tmp_path):
    store, owner = setup(tmp_path)
    owner.set_use_policy(MemoryUsePolicy(advertising=True))
    ad = ConversationMemorySession(store, user_id="u", session_id="ad", purpose="advertising")
    ad.set_use_policy(MemoryUsePolicy())
    assert not ad.get_use_policy().advertising and not ad.turns
    t = turn(1, "推荐手机。")
    seen = []
    with pytest.raises(CommitError, match="purpose"):
        ad.prepare_user(t, complete=model(value(t, "recommend"), seen), as_of=after(t))
    assert not seen


def test_low_level_proposal_obeys_purpose_and_update_policy(tmp_path):
    from memrecagent.memory_proposals import propose

    store, host = setup(tmp_path)
    t = turn(1, "以后我都更喜欢三星。")
    candidate = decode_interaction(
        json.dumps(value(t, entity="三星", relation="prefer")), turns=(t,)
    ).candidate
    args = dict(store=store, user_id="u", session_id="s", turns=(t,), people=(), as_of=after(t))
    with pytest.raises(ValueError):
        propose(candidate, purpose="advertising", **args)
    host.set_use_policy(MemoryUsePolicy(long_term_updates=False))
    assert not propose(candidate, **args).proposed_memories


@pytest.mark.parametrize("mode", ["direct", "analyst"])
def test_off_preserves_explicit_current_request_without_old_exclusion(tmp_path, mode):
    store, host = setup(tmp_path)
    host.set_use_policy(MemoryUsePolicy(False, False, False))
    t = turn(1, "这次给我看看小米。")
    p = host.prepare_user(
        t, complete=model(value(t, "recommend", entity="小米")), as_of=after(t), analysis_mode=mode
    )
    assert p.response_action == "recommend"
    assert p.memory_preview.proposal.decision.requested_entity == "小米"
    assert not p.memory_preview.proposal.decision.memories
    host.commit(p.ticket_id, as_of=after(t))
    assert store.active_memories("u", as_of=after(t))[0].relation == "exclude"


def test_scope_confirmation_cannot_commit_after_policy_change(tmp_path):
    store, host = setup(tmp_path)
    t = turn(1, "我喜欢三星，但范围未定。")
    p = host.prepare_user(
        t,
        complete=model(value(t, entity="三星", relation="prefer", scope="unclear")),
        as_of=after(t),
    )
    assert p.response_action == "clarify"
    host.set_use_policy(MemoryUsePolicy(False, False, False))
    with pytest.raises(CommitError):
        host.commit(p.ticket_id, as_of=after(t))
    assert not host.turns


def test_policy_change_during_draft_stops_before_second_stage(tmp_path):
    store, host = setup(tmp_path)
    t = turn(1, "我喜欢三星。")
    called = []
    other = ConversationMemorySession(store, user_id="u", session_id="other")

    def complete(messages):
        called.append(messages)
        other.set_use_policy(MemoryUsePolicy(False, False, False))
        return model(value(t, entity="三星", relation="prefer"))(messages)

    with pytest.raises(CommitError, match="stale"):
        host.prepare_user(t, complete=complete, as_of=after(t), analysis_mode="analyst")
    assert len(called) == 1 and not host.turns
    assert len(store.active_memories("u", as_of=after(t))) == 1


def test_updates_off_cannot_be_bypassed_by_ordinary_own_store_commit(tmp_path):
    store, host = setup(tmp_path)
    host.set_use_policy(MemoryUsePolicy(long_term_updates=False))
    t = turn(1, "我更喜欢小米，以后都优先它。")
    structured = MemoryCommitSession(store, user_id="u", session_id="structured")
    raw = value(t, entity="小米", relation="prefer")
    candidate = decode_interaction(json.dumps(raw), turns=(t,)).candidate
    p = structured.prepare(candidate, turns=(t,), as_of=after(t))
    assert not p.proposal.proposed_memories
    structured.commit(p.ticket_id, as_of=after(t))
    assert store.active_memories("u", as_of=after(t))[0].relation == "exclude"


@pytest.mark.parametrize("stage", ["direct", "draft", "final"])
@pytest.mark.parametrize("aba", [False, True])
def test_same_instance_policy_change_during_analysis_discards_old_result(tmp_path, stage, aba):
    store, host = setup(tmp_path)
    t = turn(1, "以后我都更喜欢三星。")
    raw = value(t, entity="三星", relation="prefer")
    calls = []
    mode = "direct" if stage == "direct" else "analyst"
    trigger = 2 if stage == "final" else 1

    def complete(messages):
        calls.append(messages)
        if len(calls) == trigger:
            host.set_use_policy(MemoryUsePolicy(False, False, False))
            if aba:
                host.set_use_policy(MemoryUsePolicy())
        return model(raw)(messages)

    with pytest.raises(CommitError, match="stale"):
        host.prepare_user(t, complete=complete, as_of=after(t), analysis_mode=mode)
    assert len(calls) == trigger and not host.turns
    assert len(store.active_memories("u", as_of=after(t))) == 1
