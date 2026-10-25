"""长期证据的已知跨范围混用守卫；不裁剪原文，也不改写已有快照。"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from memrecagent.memory_extraction import EvidenceSpan, ExtractionError, Turn, evidence_text

if TYPE_CHECKING:
    from memrecagent.attribute_candidates import AttributeChange
    from memrecagent.memory_extraction import CandidateAssertion

_TEMPORARY = re.compile(r"这次|本次|此次|临时|仅此一次")
_ATTRIBUTE = re.compile(
    r"预算|价[格钱]|[重轻]量|重量|续航|电池|屏幕|尺寸|英寸|存储|内存|护眼|OLED|LCD", re.I
)


def validate_persistent_evidence(
    spans: tuple[EvidenceSpan, ...],
    turns: tuple[Turn, ...],
    *,
    temporary: tuple[EvidenceSpan, ...] = (),
) -> None:
    """词面守卫只覆盖明确临时属性及已识别的本次来源，不声称读懂所有隐含语义。"""
    text = evidence_text(spans, turns)
    if any(
        _TEMPORARY.search(c) and _ATTRIBUTE.search(c) for c in re.split(r"[。；;\n]", text)
    ) or any(
        s.turn_id == t.turn_id and max(s.start, t.start) < min(s.end, t.end)
        for s in spans
        for t in temporary
    ):
        raise ExtractionError("mixed_retention_evidence")


def validate_candidate_retention(
    assertions: tuple[CandidateAssertion, ...],
    changes: tuple[AttributeChange, ...],
    turns: tuple[Turn, ...],
    *,
    temporary: tuple[EvidenceSpan, ...] = (),
) -> None:
    """已识别的他人/本次来源不能夹进本人长期证据；不替模型猜未识别语义。"""
    blocked = temporary + tuple(
        s
        for row in (*assertions, *changes)
        if row.subject.kind == "other" or row.scope in {"current", "request"}
        for s in row.evidence + row.subject.references
    )
    for row in (*assertions, *changes):
        if row.subject.kind == "self" and row.scope == "long_term":
            validate_persistent_evidence(
                row.evidence + row.subject.references, turns, temporary=blocked
            )
