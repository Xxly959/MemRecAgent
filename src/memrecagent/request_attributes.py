"""schema3本次属性候选；只读原文定位，RAM请求状态不等于长期记忆。"""

import re
from dataclasses import dataclass
from uuid import uuid4

from memrecagent.memory_commit import CommitError
from memrecagent.memory_extraction import (
    EvidenceSpan,
    ExtractionError,
    Subject,
    Turn,
    _array,
    _object,
    evidence_text,
)
from memrecagent.memory_proposals import session_person_id
from memrecagent.product_attributes import (
    NUMERIC_UNITS,
    UNITS,
    RequestConstraint,
    RequestRelaxation,
    validate_predicate,
)

DIGITS = dict(zip("零〇一二两三四五六七八九", (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9), strict=True))
SCALES = {"十": 10, "百": 100, "千": 1000, "万": 10000}


def _chinese_number(text: str) -> float:
    if "点" in text:
        whole, decimal = text.split("点")
        return _chinese_number(whole) + float("0." + "".join(str(DIGITS[c]) for c in decimal))
    if not any(c in SCALES for c in text):
        return float("".join(str(DIGITS[c]) for c in text))
    total = section = number = 0
    last_scale, tail = 1, ""
    for char in text:
        if char in DIGITS:
            number = DIGITS[char]
            tail += char
        else:
            scale = SCALES[char]
            if scale == 10000:
                total += (section + number or 1) * scale
                section = 0
            else:
                section += (number or 1) * scale
            number, last_scale, tail = 0, scale, ""
    # “两千五”是2500，“三千零五”保持3005；不对含零尾数作口语补位。
    if len(tail) == 1 and last_scale >= 100 and number:
        number *= last_scale // 10
    return float(total + section + number)


def numbers_in_text(text: str) -> set[float]:
    """仅编译原文显式数值；不能证明数值对应哪个属性或比较方向。"""
    result = set()
    for match in re.finditer(r"\d+(?:\.\d+)?(?:[kK千万])?", text):
        raw = match.group()
        scale = 1000 if raw[-1] in "kK千" else 10000 if raw[-1] == "万" else 1
        result.add(float(raw[:-1] if scale != 1 else raw) * scale)
    for match in re.finditer(r"[零〇一二两三四五六七八九十百千万点]+", text):
        try:
            result.add(_chinese_number(match.group()))
        except (KeyError, ValueError):
            continue
    return result


@dataclass(frozen=True)
class AttributeCondition:
    attribute: str
    operator: str
    value: float | str | bool
    unit: str
    evidence: tuple[EvidenceSpan, ...]


@dataclass(frozen=True)
class AttributeRequestCandidate:
    operation: str
    subject: Subject
    conditions: tuple[AttributeCondition, ...]
    remove_attributes: tuple[str, ...]
    evidence: tuple[EvidenceSpan, ...]


def decode_attribute_request(
    value: object, *, turns: tuple[Turn, ...], locate, allow_start_removals: bool = False
) -> AttributeRequestCandidate | None:
    """不接收模型提供的用户/请求ID或时间；来源由宿主原文决定。"""
    if value is None:
        return None
    row = _object(value, "operation scope subject conditions remove_attributes evidence")
    if row["operation"] not in ("start", "update", "end") or row["scope"] != "request":
        raise ExtractionError("attribute_scope")
    latest = turns[-1]

    def spans(rows, *, latest_only=True):
        checked = tuple(
            locate(turns, turn_id=r["turn_id"], quote=r["quote"], role="user")
            for r in (_object(r, "turn_id quote") for r in _array(rows))
        )
        if not checked or (latest_only and any(s.turn_id != latest.turn_id for s in checked)):
            raise ExtractionError("attribute_evidence")
        return checked

    who = _object(row["subject"], "kind references")
    kind = who["kind"]
    refs = _array(who["references"])
    if kind not in ("self", "other", "unknown") or (kind != "other" and refs):
        raise ExtractionError("attribute_subject")
    refs = spans(refs, latest_only=False) if kind == "other" else ()
    if kind == "other" and evidence_text(refs[:1], turns).strip() in {
        "他",
        "她",
        "我",
        "你",
        "它",
        "他们",
        "她们",
    }:
        raise ExtractionError("person_anchor")
    evidence = spans(row["evidence"])
    conditions = []
    for raw in _array(row["conditions"]):
        c = _object(raw, "attribute operator value unit evidence")
        try:
            validate_predicate(c["attribute"], c["operator"], c["value"], c["unit"])
        except (ValueError, TypeError, OverflowError):
            raise ExtractionError("attribute_predicate") from None
        source = spans(c["evidence"])
        text = evidence_text(source, turns)
        if (c["attribute"] in NUMERIC_UNITS and c["value"] not in numbers_in_text(text)) or (
            isinstance(c["value"], str) and c["value"].casefold() not in text.casefold()
        ):
            raise ExtractionError("attribute_value_evidence")
        conditions.append(
            AttributeCondition(c["attribute"], c["operator"], c["value"], c["unit"], source)
        )
    removed = _array(row["remove_attributes"])
    if (
        len(conditions) > 16
        or len({(c.attribute, c.operator) for c in conditions}) != len(conditions)
        or len(removed) > 16
        or any(not isinstance(a, str) or a not in UNITS for a in removed)
        or len(set(removed)) != len(removed)
        or any(c.attribute in removed for c in conditions)
    ):
        raise ExtractionError("attribute_conflict")
    if (
        (row["operation"] == "end" and (conditions or removed))
        or (row["operation"] == "update" and not conditions and not removed)
        or (row["operation"] == "start" and removed and not allow_start_removals)
    ):
        raise ExtractionError("attribute_operation")
    return AttributeRequestCandidate(
        row["operation"], Subject(kind, refs), tuple(conditions), tuple(removed), evidence
    )


@dataclass(frozen=True)
class AttributeRemoval:
    attribute: str
    evidence: tuple[EvidenceSpan, ...]


@dataclass(frozen=True)
class CurrentAttributeRequest:
    """仅RAM；源Turn快照使删除/用途重置后无法借相同轮次ID复活。"""

    request_id: str
    beneficiary_id: str
    conditions: tuple[AttributeCondition, ...]
    sources: tuple[Turn, ...]
    removals: tuple[AttributeRemoval, ...] = ()


def advance_request(
    current: CurrentAttributeRequest | None,
    candidate: AttributeRequestCandidate | None,
    *,
    user_id: str,
    session_id: str,
    turns: tuple[Turn, ...],
    beneficiary_id: str | None,
    is_recommendation: bool,
) -> CurrentAttributeRequest | None:
    """纯预计算，调用方受控commit成功后才接纳；不修改旧RAM对象。"""
    known = {t.turn_id: t for t in turns}
    if current is not None and any(known.get(t.turn_id) != t for t in current.sources):
        current = None
    if candidate is not None:
        who = candidate.subject
        person = (
            "self"
            if who.kind == "self"
            else session_person_id(user_id, session_id, who.references, turns)
            if who.kind == "other"
            else None
        )
        if person is None:
            raise CommitError("attribute_subject")
        if candidate.operation == "end":
            if current is not None and current.beneficiary_id != person:
                raise CommitError("attribute_subject")
            current = None
        else:
            if candidate.operation == "update":
                if current is not None and current.beneficiary_id != person:
                    raise CommitError("attribute_request_missing")
                if current is None and (not candidate.conditions or candidate.remove_attributes):
                    raise CommitError("attribute_request_missing")
                # 本轮完整条件可独立建立新请求；不从消失的RAM恢复旧限制或解除。
            old = current if candidate.operation == "update" else None
            conditions = {(c.attribute, c.operator): c for c in old.conditions} if old else {}
            conditions = {
                k: c
                for k, c in conditions.items()
                if c.attribute not in candidate.remove_attributes
            }
            conditions.update({(c.attribute, c.operator): c for c in candidate.conditions})
            removals = {r.attribute: r for r in old.removals} if old else {}
            for c in candidate.conditions:
                removals.pop(c.attribute, None)
            removals.update(
                {
                    a: AttributeRemoval(a, candidate.evidence + who.references)
                    for a in candidate.remove_attributes
                }
            )
            if len(conditions) > 16:
                raise CommitError("attribute_limit")
            evidence_ids = {s.turn_id for s in candidate.evidence + who.references}
            evidence_ids.update(s.turn_id for c in candidate.conditions for s in c.evidence)
            sources = {t.turn_id: t for t in old.sources} if old else {}
            sources.update({tid: known[tid] for tid in evidence_ids})
            current = CurrentAttributeRequest(
                old.request_id if old else uuid4().hex,
                person,
                tuple(conditions.values()),
                tuple(sources.values()),
                tuple(removals.values()),
            )
    if current is not None and is_recommendation and current.beneficiary_id != beneficiary_id:
        if candidate is not None and candidate.operation != "end":
            raise CommitError("attribute_subject")
        current = None
    return current


def bind_constraints(
    current: CurrentAttributeRequest | None, *, user_id: str, session_id: str
) -> tuple[RequestConstraint, ...]:
    if current is None:
        return ()
    known = {t.turn_id: t for t in current.sources}
    return tuple(
        RequestConstraint(
            user_id,
            session_id,
            current.beneficiary_id,
            c.attribute,
            c.operator,
            c.value,
            c.unit,
            ";".join(f"{session_id}:{s.turn_id}:{s.start}:{s.end}" for s in c.evidence),
            max(known[s.turn_id].available_at for s in c.evidence),
            current.request_id,
        )
        for c in current.conditions
    )


def bind_relaxations(
    current: CurrentAttributeRequest | None, *, user_id: str, session_id: str
) -> tuple[RequestRelaxation, ...]:
    """明确取消本次限制也可覆盖长期条件，结束/换人物后随RAM一并失效。"""
    if current is None:
        return ()
    known = {t.turn_id: t for t in current.sources}
    return tuple(
        RequestRelaxation(
            user_id,
            session_id,
            current.beneficiary_id,
            current.request_id,
            r.attribute,
            ";".join(f"{session_id}:{s.turn_id}:{s.start}:{s.end}" for s in r.evidence),
            max(known[s.turn_id].available_at for s in r.evidence),
        )
        for r in current.removals
    )
