"""共享一次实际草案的只读分析控制；物理调用和逻辑对照分别记账。"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Literal

from memrecagent.memory import PreferenceMemory
from memrecagent.memory_analyst import (
    AnalysisCall,
    AnalystError,
    AnalystTask,
    Complete,
    DraftInspection,
    MemoryAnalyst,
    MemoryView,
    _binding,
    _validate_heads,
    build_memory_view,
    inspect_draft,
)
from memrecagent.memory_dialogue import _candidate_semantics
from memrecagent.memory_extraction import Subject, Turn, validate_turns
from memrecagent.memory_interaction_extraction import (
    InteractionCandidate,
    build_interaction_messages,
)


@dataclass(frozen=True)
class ControlArm:
    mode: str
    status: Literal["ok", "processing_error", "provider_error", "not_run"]
    call_index: int | None
    analysis: InteractionCandidate | None = field(default=None, repr=False)
    error_code: str | None = None


@dataclass(frozen=True)
class SharedAnalysisResult:
    """direct复用calls[0]，两个最终角色各新增一条；总成本只能对calls求和。"""

    arms: tuple[ControlArm, ...]
    calls: tuple[AnalysisCall, ...]
    fatal: bool
    draft: DraftInspection | None = field(default=None, repr=False)
    memory_view: MemoryView | None = field(default=None, repr=False)


def compare_shared_draft(
    turns: tuple[Turn, ...],
    *,
    complete: Complete,
    user_id: str,
    session_id: str,
    as_of: datetime,
    snapshot_token: str,
    active_memories: tuple[PreferenceMemory, ...],
    pending_question: Turn | None = None,
    known_subjects: tuple[Subject, ...] = (),
    arm_order: tuple[str, str] = ("analyst", "analyst_blind"),
    max_calls: int = 3,
) -> SharedAnalysisResult:
    """同一草案/视图的配对；不读写磁盘、不生成提交票据、不自动重试。"""
    if (
        not callable(complete)
        or type(max_calls) is not int
        or max_calls != 3
        or not isinstance(arm_order, tuple)
        or len(arm_order) != 2
        or any(not isinstance(mode, str) for mode in arm_order)
        or set(arm_order) != {"analyst", "analyst_blind"}
    ):
        raise AnalystError("configuration")
    as_of = _binding(user_id, session_id, as_of, snapshot_token)
    validate_turns(turns)
    if not isinstance(turns, tuple) or any(t.available_at >= as_of for t in turns):
        raise AnalystError("task")
    _validate_heads(active_memories, user_id=user_id, as_of=as_of)
    # 预先验证最新用户、人物及实际相邻问题，不能发送后才发现任务本身不合法。
    build_interaction_messages(
        turns, (), pending_question=pending_question, known_subjects=known_subjects
    )
    calls: list[AnalysisCall] = []
    arms = [ControlArm(mode, "not_run", None) for mode in ("direct", *arm_order)]
    try:
        draft = inspect_draft(
            turns,
            complete=complete,
            pending_question=pending_question,
            known_subjects=known_subjects,
        )
    except AnalystError as error:
        if error.call is not None:
            calls.append(error.call)
        arms[0] = ControlArm(
            "direct",
            error.call.status if error.call else "processing_error",
            0 if calls else None,
            error_code=error.code,
        )
        return SharedAnalysisResult(tuple(arms), tuple(calls), True)
    calls.append(draft.call)
    arms[0] = ControlArm("direct", draft.call.status, 0, draft.analysis, draft.error_code)
    view = (
        MemoryView(user_id, session_id, as_of, snapshot_token, (), False)
        if draft.analysis is None
        else build_memory_view(
            user_id=user_id,
            session_id=session_id,
            as_of=as_of,
            snapshot_token=snapshot_token,
            active_memories=active_memories,
            draft=draft.analysis,
        )
    )
    for index, mode in enumerate(arm_order, 1):
        task = AnalystTask(
            user_id,
            session_id,
            as_of,
            turns,
            draft.raw,
            view,
            pending_question,
            known_subjects,
            draft.error_code,
            expose_draft=mode == "analyst",
        )
        try:
            final = MemoryAnalyst(complete).analyze(task)
        except AnalystError as error:
            physical_index = len(calls) if error.call is not None else None
            if error.call is not None:
                calls.append(error.call)
            arms[index] = ControlArm(
                mode,
                error.call.status if error.call else "processing_error",
                physical_index,
                error_code=error.code,
            )
            if (
                error.call is None
                or error.call.status == "provider_error"
                or error.code in {"reply_type", "reply_metadata", "output_size"}
            ):
                return SharedAnalysisResult(tuple(arms), tuple(calls), True, draft, view)
        else:
            calls.append(final.call)
            arms[index] = ControlArm(mode, "ok", len(calls) - 1, final.analysis)
    return SharedAnalysisResult(tuple(arms), tuple(calls), False, draft, view)


def analysis_projection(analysis: InteractionCandidate, turns: tuple[Turn, ...]) -> dict:
    """已严格解码的候选语义；不按引用措辞计分，仍保留实际人物锚点及属性。"""
    known = validate_turns(turns)

    def text(span):
        return known[span.turn_id].content[span.start : span.end].strip()

    def row(value):
        data = asdict(value)
        data.pop("evidence", None)
        data.update(
            subject=value.subject.kind,
            person=text(value.subject.references[0]) if value.subject.kind == "other" else None,
        )
        return data

    def ordered(values):
        return sorted(values, key=lambda v: json.dumps(v, sort_keys=True, ensure_ascii=False))

    attributes = analysis.attribute_request
    request = None
    if attributes is not None:
        request = row(attributes)
        request["conditions"] = ordered(
            [{k: v for k, v in asdict(c).items() if k != "evidence"} for c in attributes.conditions]
        )
        request["remove_attributes"] = sorted(attributes.remove_attributes)
    confirmation = None
    if analysis.confirmation is not None:
        c = analysis.confirmation
        confirmation = dict(
            question_turn_id=c.question_turn_id,
            outcome=c.choice.outcome,
            subject=c.choice.subject,
            scope=c.choice.scope,
            person=text(c.choice.person_reference)
            if c.choice.person_reference is not None
            else None,
        )
    return _candidate_semantics(analysis.candidate, turns) | dict(
        intents=sorted(analysis.intents),
        clues=ordered(row(c) for c in analysis.clues),
        unresolved=sorted(u.field for u in analysis.candidate.unresolved),
        attribute_request=request,
        attribute_changes=ordered(row(c) for c in analysis.attribute_changes),
        confirmation=confirmation,
    )
