"""用合成证据验证人物、态度与时间能无损保存，非法记录不能进入存储。"""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta, timezone

import pytest

from memrecagent.memory import PreferenceMemory

T0 = datetime(2026, 10, 20, 1, tzinfo=UTC)


def sample_memory(**changes):
    values = dict(
        memory_id="m1",
        user_id="demo-user",
        category="phone",
        entity="小米",
        relation="exclude",
        evidence_ref="S1:T3",
        evidence_text='以后买手机别推荐小米\n这是合成表达，不是本人事实。{"例子": true}',
        occurred_at=T0,
        available_at=T0 + timedelta(minutes=1),
        confidence=None,
    )
    return PreferenceMemory(**(values | changes))


def test_roundtrip_preserves_evidence_subject_and_unknown_confidence():
    memory = sample_memory()
    data = memory.to_dict()
    assert data["schema_version"] == 1
    assert data["subject_id"] == "self"
    assert data["confidence"] is None
    assert data["occurred_at"] == "2026-10-20T01:00:00+00:00"
    assert data["evidence_text"] == memory.evidence_text
    assert PreferenceMemory.from_dict(data) == memory


def test_revision_is_new_evidence_instead_of_mutating_the_old_snapshot():
    old = sample_memory()
    new = replace(old, memory_id="m2", relation="neutral", supersedes="m1")
    assert old.relation == "exclude" and old.supersedes is None
    assert new.relation == "neutral" and new.supersedes == "m1"
    with pytest.raises(FrozenInstanceError):
        old.relation = "prefer"


@pytest.mark.parametrize(
    "changes",
    [
        {"user_id": " "},
        {"user_id": "u" * 129},
        {"subject_id": None},
        {"entity": 1},
        {"category": ""},
        {"memory_id": ""},
        {"evidence_ref": ""},
        {"evidence_text": " "},
        {"relation": "question"},
        {"confidence": True},
        {"confidence": float("nan")},
        {"confidence": float("inf")},
        {"confidence": -0.1},
        {"confidence": 1.1},
        {"occurred_at": T0.replace(tzinfo=None)},
        {"available_at": T0 - timedelta(seconds=1)},
        {"supersedes": "m1"},
        {"supersedes": ""},
    ],
)
def test_invalid_memory_is_rejected_without_echoing_identity_or_evidence(changes):
    with pytest.raises(ValueError) as error:
        sample_memory(**changes)
    assert "demo-user" not in str(error.value)
    assert "以后买手机" not in str(error.value)


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"unexpected": "secret"},
        {"confidence": "high"},
        {"occurred_at": "bad-private-text"},
    ],
)
def test_invalid_serialized_fields_or_versions_are_rejected(changes):
    data = sample_memory().to_dict() | changes
    with pytest.raises(ValueError) as error:
        PreferenceMemory.from_dict(data)
    assert "bad-private-text" not in str(error.value)


def test_missing_serialized_fields_are_not_silently_filled():
    data = sample_memory().to_dict()
    del data["subject_id"]
    with pytest.raises(ValueError):
        PreferenceMemory.from_dict(data)


def test_equivalent_time_offsets_serialize_to_utc():
    memory = sample_memory(occurred_at=T0.astimezone(timezone(timedelta(hours=8))))
    assert memory.to_dict()["occurred_at"] == "2026-10-20T01:00:00+00:00"


def test_other_person_preference_cannot_be_a_persistent_user_memory():
    with pytest.raises(ValueError, match="subject_id"):
        sample_memory(subject_id="relative:uncle")
