"""只读属性变更候选及宿主编译；模型不提供身份、时间或修订父节点。"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from hashlib import sha256

from memrecagent.attribute_memory import AttributeMemory
from memrecagent.memory_evidence_policy import (
    validate_candidate_retention,
    validate_persistent_evidence,
)
from memrecagent.memory_extraction import (
    EvidenceSpan,
    ExtractionError,
    Subject,
    Turn,
    _array,
    _object,
    evidence_text,
)
from memrecagent.memory_store import JsonlMemoryStore
from memrecagent.product_attributes import NUMERIC_UNITS, UNITS, validate_predicate


@dataclass(frozen=True)
class AttributeChange:
    subject: Subject
    category: str
    attribute: str
    operator: str
    value: float | str | bool | None = field(repr=False)
    unit: str
    is_hard: bool
    active: bool
    scope: str
    evidence: tuple[EvidenceSpan, ...] = field(repr=False)


@dataclass(frozen=True)
class ContextAttributePreference:
    beneficiary_id: str
    category: str
    attribute: str
    operator: str
    value: float | str | bool | None = field(repr=False)
    unit: str
    is_hard: bool
    active: bool
    evidence_ref: str = field(repr=False)
    available_at: datetime
    retention: str = "session"


def decode_attribute_changes(
    value: object, *, turns: tuple[Turn, ...], locate
) -> tuple[AttributeChange, ...]:
    """来源必须包含本轮用户表达；撤销不要求模型从记忆抄旧数值。"""
    from memrecagent.request_attributes import numbers_in_text

    rows = _array(value)
    if len(rows) > 16:
        raise ExtractionError("attribute_limit")
    result = []
    for value in rows:
        row = _object(
            value, "subject category attribute operator value unit is_hard active scope evidence"
        )
        if (
            row["category"] != "phone"
            or row["scope"] not in ("long_term", "request", "unclear")
            or type(row["is_hard"]) is not bool
            or type(row["active"]) is not bool
            or (row["scope"] == "request" and row["is_hard"])
        ):
            raise ExtractionError("attribute_scope")
        who = _object(row["subject"], "kind references")
        refs = tuple(
            locate(turns, turn_id=q["turn_id"], quote=q["quote"], role="user")
            for q in (_object(q, "turn_id quote") for q in _array(who["references"]))
        )
        if who["kind"] not in ("self", "other", "unknown") or (who["kind"] == "other") != bool(
            refs
        ):
            raise ExtractionError("attribute_subject")
        if refs and evidence_text(refs[:1], turns).strip() in {
            "他",
            "她",
            "我",
            "你",
            "它",
            "他们",
            "她们",
        }:
            raise ExtractionError("person_anchor")
        spans = tuple(
            locate(turns, turn_id=q["turn_id"], quote=q["quote"], role="user")
            for q in (_object(q, "turn_id quote") for q in _array(row["evidence"]))
        )
        if not spans or any(s.turn_id != turns[-1].turn_id for s in spans):
            raise ExtractionError("attribute_evidence")
        a = row["attribute"]
        if not isinstance(a, str) or a not in UNITS:
            raise ExtractionError("attribute_predicate")
        if not row["active"] and row["value"] is not None:
            raise ExtractionError("attribute_retraction")
        probe = (
            row["value"]
            if row["active"]
            else (1 if a in NUMERIC_UNITS else True if a == "eye_care" else "type")
        )
        try:
            validate_predicate(a, row["operator"], probe, row["unit"])
        except (ValueError, TypeError, OverflowError):
            raise ExtractionError("attribute_predicate") from None
        text = evidence_text(spans, turns)
        if row["active"] and (
            (a in NUMERIC_UNITS and row["value"] not in numbers_in_text(text))
            or (isinstance(row["value"], str) and row["value"].casefold() not in text.casefold())
        ):
            raise ExtractionError("attribute_value_evidence")
        if row["scope"] != "request":
            validate_persistent_evidence(spans, turns)
        result.append(
            AttributeChange(
                Subject(who["kind"], refs),
                row["category"],
                a,
                row["operator"],
                row["value"],
                row["unit"],
                row["is_hard"],
                row["active"],
                row["scope"],
                spans,
            )
        )
    keys = [(r.subject, r.category, r.attribute, r.operator, r.is_hard, r.scope) for r in result]
    if len(set(keys)) != len(keys):
        raise ExtractionError("attribute_conflict")
    return tuple(result)


def compile_attribute_changes(
    changes: tuple[AttributeChange, ...],
    *,
    store: JsonlMemoryStore,
    user_id: str,
    session_id: str,
    turns: tuple[Turn, ...],
    as_of: datetime,
) -> tuple[tuple[AttributeMemory, ...], tuple[ContextAttributePreference, ...], bool]:
    """宿主读取当前头决定修订；任何不明阻止整批，不拿已知一半先写盘。"""
    from memrecagent.memory_interaction_extraction import locate_quote
    from memrecagent.memory_proposals import session_person_id

    if not changes:
        return (), (), False
    if not isinstance(changes, tuple) or any(not isinstance(c, AttributeChange) for c in changes):
        raise ExtractionError("attribute_candidate")
    # dataclass不是信任边界；公开宿主入口也必须重新经过来源/数值/字段编译。
    source_turns = {t.turn_id: t for t in turns}

    def quote(s):
        if (
            not isinstance(s, EvidenceSpan)
            or s.turn_id not in source_turns
            or type(s.start) is not int
            or type(s.end) is not int
            or not 0 <= s.start < s.end <= len(source_turns[s.turn_id].content)
        ):
            raise ExtractionError("attribute_evidence")
        return {"turn_id": s.turn_id, "quote": source_turns[s.turn_id].content[s.start : s.end]}

    rows = []
    for c in changes:
        row = asdict(c)
        row["subject"] = {
            "kind": c.subject.kind,
            "references": [quote(s) for s in c.subject.references],
        }
        row["evidence"] = [quote(s) for s in c.evidence]
        rows.append(row)
    changes = decode_attribute_changes(rows, turns=turns, locate=locate_quote)
    validate_candidate_retention((), changes, turns)
    unresolved = any(c.subject.kind == "unknown" or c.scope == "unclear" for c in changes)
    if unresolved:
        return (), (), True
    known = {t.turn_id: t for t in turns}
    heads = {
        (m.category, m.attribute, m.operator, m.is_hard): m
        for m in store.active_attribute_memories(user_id, as_of=as_of, include_inactive=True)
    }
    history = {m.memory_id for m in store.list_attribute_memories(user_id, as_of=as_of)}
    records, contexts = [], []
    for c in changes:
        sources = tuple(dict.fromkeys(c.evidence + c.subject.references))
        if any(
            s.turn_id not in known
            or known[s.turn_id].role != "user"
            or not 0 <= s.start < s.end <= len(known[s.turn_id].content)
            for s in sources
        ):
            raise ExtractionError("attribute_evidence")
        occurred = max(known[s.turn_id].occurred_at for s in c.evidence)
        available = max(known[s.turn_id].available_at for s in sources)
        if max(occurred, available) >= as_of:
            raise ExtractionError("future_evidence")
        ref = ";".join(f"{session_id}:{s.turn_id}[{s.start}:{s.end}]" for s in sources)
        if c.scope == "request":
            contexts.append(
                ContextAttributePreference(
                    "self"
                    if c.subject.kind == "self"
                    else session_person_id(user_id, session_id, c.subject.references, turns),
                    c.category,
                    c.attribute,
                    c.operator,
                    c.value,
                    c.unit,
                    False,
                    c.active,
                    ref,
                    available,
                    "request",
                )
            )
            continue
        if c.subject.kind == "other":
            contexts.append(
                ContextAttributePreference(
                    session_person_id(user_id, session_id, c.subject.references, turns),
                    c.category,
                    c.attribute,
                    c.operator,
                    c.value,
                    c.unit,
                    c.is_hard,
                    c.active,
                    ref,
                    available,
                )
            )
            continue
        key = (c.category, c.attribute, c.operator, c.is_hard)
        old = heads.get(key)
        memory_id = (
            "attribute:"
            + sha256(
                json.dumps(
                    [user_id, session_id, asdict(c)], ensure_ascii=False, sort_keys=True
                ).encode()
            ).hexdigest()
        )
        if memory_id in history or (not c.active and (old is None or not old.active)):
            continue
        # 本轮明确要求可用于当前推荐，写入/读档权限是另一条边界。
        # self只供本轮使用，不发布成持续会话状态。
        if c.active:
            contexts.append(
                ContextAttributePreference(
                    "self",
                    c.category,
                    c.attribute,
                    c.operator,
                    c.value,
                    c.unit,
                    c.is_hard,
                    True,
                    ref,
                    available,
                    "turn",
                )
            )
        records.append(
            AttributeMemory(
                memory_id,
                user_id,
                c.category,
                c.attribute,
                c.operator,
                c.value if c.active else old.value,
                c.unit,
                c.is_hard,
                c.active,
                ref,
                evidence_text(sources, turns),
                occurred,
                available,
                supersedes=old.memory_id if old else None,
            )
        )
    return tuple(records), tuple(contexts), False
