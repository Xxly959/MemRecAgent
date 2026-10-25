"""真实文件验证同轮多条提交的原子性和删除水位版本检查。"""

from dataclasses import replace
from datetime import timedelta

import pytest
from test_memory import T0, sample_memory
from test_memory_store import AFTER, file_for

from memrecagent.memory_store import JsonlMemoryStore


def test_batch_tail_invalid_does_not_publish_earlier_valid_row(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    store.add(sample_memory())
    before = file_for(tmp_path).read_bytes()
    with pytest.raises(ValueError):
        store.add_many((sample_memory(memory_id="b", entity="三星"), sample_memory()))
    assert file_for(tmp_path).read_bytes() == before


def test_snapshot_includes_deletion_even_when_visible_history_stays_empty(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    old = store.snapshot_token("demo-user")
    store.delete_entity("demo-user", category="phone", entity="小米", deleted_at=AFTER)
    new = store.snapshot_token("demo-user")
    assert old != new
    assert store.list_memories("demo-user", as_of=AFTER) == ()
    assert JsonlMemoryStore(tmp_path).snapshot_token("demo-user") == new


def test_stale_token_and_cross_owner_batch_do_not_create_file(tmp_path):
    store = JsonlMemoryStore(tmp_path)
    token = store.snapshot_token("demo-user")
    with pytest.raises(ValueError):
        store.add_many((sample_memory(), sample_memory(user_id="another-user")))
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []
    store.add(sample_memory(memory_id="b", entity="三星"))
    before = file_for(tmp_path).read_bytes()
    with pytest.raises(ValueError):
        store.add_many((sample_memory(),), expected_token=token)
    assert file_for(tmp_path).read_bytes() == before


def test_valid_batch_persists_once_and_publication_failure_preserves_bytes(tmp_path, monkeypatch):
    store = JsonlMemoryStore(tmp_path)
    first = sample_memory()
    second = replace(
        first,
        memory_id="m2",
        supersedes="m1",
        relation="neutral",
        occurred_at=T0 + timedelta(hours=1),
        available_at=T0 + timedelta(hours=1),
    )
    store.add_many((first, second))
    assert JsonlMemoryStore(tmp_path).active_memories("demo-user", as_of=AFTER) == (second,)
    before = file_for(tmp_path).read_bytes()

    def fail(*args):
        raise OSError("合成发布失败")

    monkeypatch.setattr("memrecagent.memory_store.os.replace", fail)
    with pytest.raises(OSError):
        store.add_many((sample_memory(memory_id="b", entity="三星"),))
    assert file_for(tmp_path).read_bytes() == before
    assert list(tmp_path.iterdir()) == [file_for(tmp_path)]
