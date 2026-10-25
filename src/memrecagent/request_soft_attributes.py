"""临时软属性依附当前选购任务；纯预计算，提交成功后才发布RAM。"""

from dataclasses import replace
from uuid import uuid4

from memrecagent.attribute_candidates import ContextAttributePreference
from memrecagent.memory_commit import CommitError
from memrecagent.memory_extraction import Turn
from memrecagent.request_attributes import AttributeRequestCandidate, CurrentAttributeRequest


def advance_soft_request(
    previous: CurrentAttributeRequest | None,
    current: CurrentAttributeRequest | None,
    preferences: tuple[ContextAttributePreference, ...],
    changes: tuple[ContextAttributePreference, ...],
    *,
    candidate: AttributeRequestCandidate | None,
    turns: tuple[Turn, ...],
    beneficiary_id: str | None,
    is_recommendation: bool,
) -> tuple[CurrentAttributeRequest | None, tuple[ContextAttributePreference, ...]]:
    """同任务合并，start/end/换人清理；只接收宿主编译后的候选。"""
    known = {t.turn_id: t for t in turns}
    if (
        previous is None
        or current is None
        or previous.request_id != current.request_id
        or any(known.get(t.turn_id) != t for t in previous.sources)
    ):
        preferences = ()
    incoming = tuple(c for c in changes if c.retention == "request")
    if incoming:
        people = {c.beneficiary_id for c in incoming}
        if len(people) != 1 or (candidate is not None and candidate.operation == "end"):
            raise CommitError("attribute_subject")
        person = incoming[0].beneficiary_id
        if is_recommendation and person != beneficiary_id:
            raise CommitError("attribute_subject")
        if current is not None and current.beneficiary_id != person:
            if candidate is not None:
                raise CommitError("attribute_subject")
            current, preferences = None, ()
        if current is None:
            current = CurrentAttributeRequest(uuid4().hex, person, (), ())
        # 临时证据也保留原Turn快照，fresh重置后不可只靠相同轮次ID复活。
        sources = {t.turn_id: t for t in current.sources}
        sources[turns[-1].turn_id] = turns[-1]
        # 人物指代可引用早先轮次；整个实际上下文仍受宿主长度上限约束。
        sources.update(known)
        current = replace(current, sources=tuple(sources.values()))
    if current is None:
        return None, ()
    # 后来的长期纠正不能被更早的临时同属性偏好重新覆盖。
    updated = {
        (c.attribute, c.operator): c
        for c in preferences
        if not any(
            n.retention != "request"
            and n.beneficiary_id == c.beneficiary_id
            and n.attribute == c.attribute
            for n in changes
        )
    }
    for c in incoming:
        updated = {
            k: old
            for k, old in updated.items()
            if old.attribute != c.attribute or (old.active and c.active)
        }
        updated[c.attribute, c.operator] = c
    if len(updated) > 16:
        raise CommitError("attribute_limit")
    return current, tuple(updated.values())
