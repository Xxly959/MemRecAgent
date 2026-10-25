"""实际临时文件验证本人偏好修订、跨重启读取与失败时证据不丢失。"""

import json
from dataclasses import replace
from datetime import timedelta
from hashlib import sha256

import pytest
from test_memory import T0, sample_memory

from memrecagent.memory_store import JsonlMemoryStore

AFTER = T0 + timedelta(days=1)


def file_for(root, user="demo-user"):
    return root / (sha256(user.encode()).hexdigest() + ".jsonl")


def revision(old, **changes):
    return replace(
        old,
        memory_id="m2",
        relation="neutral",
        supersedes=old.memory_id,
        occurred_at=T0 + timedelta(hours=1),
        available_at=T0 + timedelta(hours=2),
        evidence_ref="S2:T1",
        evidence_text="以后可以推荐，但不代表喜欢",
        **changes,
    )


def test_evidence_survives_reopening_and_users_and_categories_are_isolated(tmp_path):
    root = tmp_path / "memory"
    store = JsonlMemoryStore(root)
    memory = sample_memory()
    store.add(memory)
    reopened = JsonlMemoryStore(root)
    assert reopened.list_memories("demo-user", as_of=AFTER) == (memory,)
    assert reopened.list_memories("another-user", as_of=AFTER) == ()
    assert reopened.list_memories("demo-user", category="instrument", as_of=AFTER) == ()
    assert len(file_for(root).read_text().splitlines()) == 1


def test_only_explicit_revision_changes_active_preference_and_preserves_old_evidence(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    old = sample_memory()
    new = revision(old)
    store.add(old)
    store.add(new)
    assert store.list_memories("demo-user", as_of=AFTER) == (old, new)
    assert store.active_memories("demo-user", as_of=AFTER) == (new,)
    assert store.active_memories("demo-user", as_of=T0 + timedelta(hours=1)) == (old,)
    assert store.active_memories("demo-user", as_of=T0 + timedelta(hours=2)) == (old,)


def test_time_boundary_and_late_evidence_never_leak_into_the_past(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    memory = sample_memory(available_at=T0 + timedelta(hours=4))
    store.add(memory)
    for time in [T0, T0 + timedelta(hours=3), T0 + timedelta(hours=4)]:
        assert store.list_memories("demo-user", as_of=time) == ()
    assert store.list_memories("demo-user", as_of=AFTER) == (memory,)


@pytest.mark.parametrize(
    "case",
    [
        "duplicate_id",
        "unlinked",
        "missing_parent",
        "wrong_entity",
        "old_event",
        "earlier_available",
        "fork",
    ],
)
def test_invalid_revision_cannot_overwrite_existing_evidence(tmp_path, case):
    store = JsonlMemoryStore(tmp_path)
    old = sample_memory()
    store.add(old)
    new = revision(old)
    candidate = {
        "duplicate_id": old,
        "unlinked": replace(new, supersedes=None),
        "missing_parent": replace(new, supersedes="not-found"),
        "wrong_entity": replace(new, entity="其他品牌"),
        "old_event": replace(new, occurred_at=T0),
        "earlier_available": replace(new, available_at=T0 + timedelta(hours=1)),
        "fork": replace(new, memory_id="m3"),
    }[case]
    if case == "earlier_available":
        # 旧证据直到3小时才可用，不能被2小时的修订引用。
        root = tmp_path / "late"
        store = JsonlMemoryStore(root)
        store.add(replace(old, available_at=T0 + timedelta(hours=3)))
    else:
        root = tmp_path
    if case == "fork":
        store.add(new)
    before = file_for(root).read_bytes()
    with pytest.raises(ValueError):
        store.add(candidate)
    assert file_for(root).read_bytes() == before


@pytest.mark.parametrize(
    "bad", [b'{"private-content":', b"\xff", b"{}", b'{"schema_version":1,"schema_version":2}']
)
def test_bad_trailing_line_is_not_hidden_by_time_or_category_filters(tmp_path, bad):
    store = JsonlMemoryStore(tmp_path)
    store.add(sample_memory())
    with file_for(tmp_path).open("ab") as stream:
        stream.write(bad + b"\n")
    before = file_for(tmp_path).read_bytes()
    for operation in [
        lambda: store.list_memories("demo-user", category="other", as_of=T0),
        lambda: store.add(sample_memory(memory_id="another", entity="其他品牌")),
    ]:
        with pytest.raises(ValueError) as error:
            operation()
        assert "demo-user" not in str(error.value)
        assert "private-content" not in str(error.value)
        assert "以后买手机" not in str(error.value)
    assert file_for(tmp_path).read_bytes() == before


@pytest.mark.parametrize("mutation", ["owner", "schema", "duplicate"])
def test_tampered_owner_version_or_duplicate_id_rejects_whole_file(tmp_path, mutation):
    data = sample_memory().to_dict()
    if mutation == "owner":
        data["user_id"] = "someone-else"
    if mutation == "schema":
        data["schema_version"] = 99
    text = json.dumps(data) + "\n"
    file_for(tmp_path).write_text(text * (2 if mutation == "duplicate" else 1))
    with pytest.raises(ValueError):
        JsonlMemoryStore(tmp_path).list_memories("demo-user", as_of=AFTER)


def test_failed_atomic_publish_keeps_old_bytes_and_removes_own_temporary_file(
    tmp_path, monkeypatch
):
    store = JsonlMemoryStore(tmp_path)
    old = sample_memory()
    store.add(old)
    before = file_for(tmp_path).read_bytes()

    def fail_publish(*args):
        raise OSError("模拟发布失败")

    monkeypatch.setattr("memrecagent.memory_store.os.replace", fail_publish)
    with pytest.raises(OSError):
        store.add(revision(old))
    assert file_for(tmp_path).read_bytes() == before
    assert list(tmp_path.iterdir()) == [file_for(tmp_path)]


def test_path_like_user_identifier_does_not_escape_storage_root(tmp_path):
    root = tmp_path / "memory"
    user = "../outside"
    memory = sample_memory(user_id=user)
    store = JsonlMemoryStore(root)
    store.add(memory)
    assert store.list_memories(user, as_of=AFTER) == (memory,)
    assert list(tmp_path.iterdir()) == [root]
    assert len(list(root.iterdir())) == 1


def test_missing_user_is_empty_but_invalid_query_is_rejected(tmp_path):
    store = JsonlMemoryStore(tmp_path / "absent")
    assert store.list_memories("new-user", as_of=AFTER) == ()
    with pytest.raises(ValueError):
        store.list_memories("new-user", as_of=AFTER.replace(tzinfo=None))
    with pytest.raises(ValueError):
        store.list_memories(" ", as_of=AFTER)
