"""严格解码带原文证据的模型候选；字段合法不等于语义正确。"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from memrecagent.memory import _text, _utc


class ExtractionError(ValueError):
    """只暴露错误码，不回显对话、凭证或模型响应。"""

    def __init__(self, code: str):
        self.code = code
        super().__init__(f"偏好提取失败：{code}")


@dataclass(frozen=True)
class Turn:
    """调用方提供的可信轮次；模型不能指定身份或时间。"""

    turn_id: str
    role: Literal["user", "assistant"]
    content: str = field(repr=False)
    occurred_at: datetime
    available_at: datetime

    def __post_init__(self) -> None:
        _text(self.turn_id, "turn_id")
        _text(self.content, "content")
        if self.role not in {"user", "assistant"}:
            raise ValueError("轮次角色无效")
        for name in ("occurred_at", "available_at"):
            object.__setattr__(self, name, _utc(getattr(self, name), name))
        if self.available_at < self.occurred_at:
            raise ValueError("轮次可用时间早于发生时间")


@dataclass(frozen=True)
class EvidenceSpan:
    turn_id: str
    start: int
    end: int


@dataclass(frozen=True)
class Subject:
    kind: Literal["self", "other", "unknown"]
    references: tuple[EvidenceSpan, ...]


@dataclass(frozen=True)
class CandidateRequest:
    is_recommendation: bool
    beneficiary: Subject
    category: str | None
    requested_entity: str | None
    change_scope: Literal["request", "long_term", "unclear"]
    evidence: tuple[EvidenceSpan, ...]


@dataclass(frozen=True)
class CandidateAssertion:
    subject: Subject
    category: str
    entity: str
    relation: Literal["prefer", "dislike", "exclude", "neutral", "none"]
    scope: Literal["long_term", "current", "unclear"]
    evidence: tuple[EvidenceSpan, ...]


@dataclass(frozen=True)
class Unresolved:
    field: Literal["subject", "scope", "entity", "relation"]
    evidence: tuple[EvidenceSpan, ...]


@dataclass(frozen=True)
class ExtractionCandidate:
    request: CandidateRequest
    assertions: tuple[CandidateAssertion, ...]
    unresolved: tuple[Unresolved, ...]


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    """重复键不能静默覆盖；也供供应商响应边界复用。"""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ExtractionError("duplicate_key")
        result[key] = value
    return result


def _object(value: object, keys: str) -> dict:
    if not isinstance(value, dict) or set(value) != set(keys.split()):
        raise ExtractionError("fields")
    return value


def _string(value: object, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ExtractionError("text")
    return value


def _enum(value: object, allowed: str) -> str:
    if not isinstance(value, str) or value not in allowed.split():
        raise ExtractionError("enum")
    return value


def _array(value: object) -> list:
    if not isinstance(value, list) or len(value) > 32:
        raise ExtractionError("array")
    return value


def validate_turns(turns: tuple[Turn, ...]) -> dict[str, Turn]:
    """验证可信上下文有序且唯一，不替调用方猜测缺失轮次。"""
    if not turns or len(turns) > 128 or any(not isinstance(t, Turn) for t in turns):
        raise ExtractionError("turns")
    if len({t.turn_id for t in turns}) != len(turns):
        raise ExtractionError("turns")
    if any(
        b.occurred_at < a.occurred_at or b.available_at < a.available_at
        for a, b in zip(turns, turns[1:], strict=False)
    ):
        raise ExtractionError("turn_order")
    if sum(len(t.content) for t in turns) > 32000:
        raise ExtractionError("input_size")
    return {t.turn_id: t for t in turns}


def evidence_text(spans: tuple[EvidenceSpan, ...], turns: tuple[Turn, ...]) -> str:
    """只切取已核验的原话，跨片段用换行分隔，不使用模型摘要。"""
    known = {t.turn_id: t for t in turns}
    return "\n".join(known[s.turn_id].content[s.start : s.end] for s in spans)


def decode_candidate(raw: str, *, turns: tuple[Turn, ...]) -> ExtractionCandidate:
    """拒绝额外字段、伪造证据和不完整输出；不执行任何写入。"""
    known = validate_turns(turns)
    try:
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > 65536:
            raise ExtractionError("output_size")
        data = json.loads(
            raw,
            object_pairs_hook=unique_object,
            parse_constant=lambda _: (_ for _ in ()).throw(ExtractionError("json")),
        )
        _object(data, "request assertions unresolved")

        def spans(value: object, *, required: bool = True) -> tuple[EvidenceSpan, ...]:
            result = []
            for item in _array(value):
                row = _object(item, "turn_id start end")
                turn_id = _string(row["turn_id"])
                start, end = row["start"], row["end"]
                turn = known.get(turn_id)
                if (
                    turn is None
                    or turn.role != "user"
                    or type(start) is not int
                    or type(end) is not int
                    or not 0 <= start < end <= len(turn.content)
                ):
                    raise ExtractionError("evidence")
                result.append(EvidenceSpan(turn_id, start, end))
            if required and not result:
                raise ExtractionError("evidence")
            return tuple(result)

        def subject(value: object) -> Subject:
            row = _object(value, "kind references")
            kind = _enum(row["kind"], "self other unknown")
            references = spans(row["references"], required=kind == "other")
            if kind == "other":
                anchor = evidence_text(references[:1], turns).strip()
                if anchor in {"他", "她", "他们", "她们", "我", "你", "您", "它", "它们"}:
                    raise ExtractionError("person_anchor")
            return Subject(kind, references)

        row = _object(
            data["request"],
            "is_recommendation beneficiary category requested_entity change_scope evidence",
        )
        if type(row["is_recommendation"]) is not bool:
            raise ExtractionError("boolean")
        request = CandidateRequest(
            row["is_recommendation"],
            subject(row["beneficiary"]),
            _string(row["category"], nullable=True),
            _string(row["requested_entity"], nullable=True),
            _enum(row["change_scope"], "request long_term unclear"),
            spans(row["evidence"], required=row["is_recommendation"]),
        )
        if request.requested_entity and request.requested_entity not in evidence_text(
            request.evidence, turns
        ):
            raise ExtractionError("entity_evidence")
        assertions = []
        for item in _array(data["assertions"]):
            row = _object(item, "subject category entity relation scope evidence")
            evidence = spans(row["evidence"])
            entity = _string(row["entity"])
            if entity not in evidence_text(evidence, turns):
                raise ExtractionError("entity_evidence")
            assertions.append(
                CandidateAssertion(
                    subject(row["subject"]),
                    _string(row["category"]),
                    entity,
                    _enum(row["relation"], "prefer dislike exclude neutral none"),
                    _enum(row["scope"], "long_term current unclear"),
                    evidence,
                )
            )
        unresolved = []
        for item in _array(data["unresolved"]):
            row = _object(item, "field evidence")
            unresolved.append(
                Unresolved(
                    _enum(row["field"], "subject scope entity relation"), spans(row["evidence"])
                )
            )
        return ExtractionCandidate(request, tuple(assertions), tuple(unresolved))
    except ExtractionError:
        raise
    except (ValueError, TypeError, RecursionError, UnicodeError):
        raise ExtractionError("json") from None
