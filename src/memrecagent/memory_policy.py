"""基于已解析人物和意图的保守策略；不声称识别任意自然语言。"""

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Literal

from memrecagent.memory import PreferenceMemory, _memory_age, _text, _utc
from memrecagent.memory_store import JsonlMemoryStore
from memrecagent.memory_use_policy import JsonMemoryUseStore, _check_purpose


@dataclass(frozen=True)
class ContextPreference:
    """只随调用方的当前会话传入，包含他人喜好，不能作为长期记录写盘。"""

    user_id: str = field(repr=False)
    session_id: str = field(repr=False)
    beneficiary_id: str | None
    category: str
    entity: str
    relation: Literal["prefer", "dislike", "exclude", "neutral"]
    evidence_ref: str = field(repr=False)
    evidence_text: str = field(repr=False)
    available_at: datetime
    retention: Literal["request", "session"] = "request"

    def __post_init__(self) -> None:
        for name in [
            "user_id",
            "session_id",
            "category",
            "entity",
            "relation",
            "evidence_ref",
            "evidence_text",
        ]:
            _text(getattr(self, name), name)
        if len(self.user_id) > 128:
            raise ValueError("user_id: 长度超过128字符")
        if self.beneficiary_id is not None:
            _text(self.beneficiary_id, "beneficiary_id")
        if self.relation not in {"prefer", "dislike", "exclude", "neutral"}:
            raise ValueError("relation: 不支持的上下文关系")
        if self.retention not in {"request", "session"}:
            raise ValueError("retention: 不支持的上下文保留范围")
        object.__setattr__(self, "available_at", _utc(self.available_at, "available_at"))


@dataclass(frozen=True)
class RecommendationDecision:
    """明确是否澄清、使用哪些来源；被临时覆盖的证据仍留在存储中。"""

    status: Literal["ready", "clarify"]
    beneficiary_id: str | None
    requested_entity: str | None
    memories: tuple[PreferenceMemory, ...] = ()
    context_preferences: tuple[ContextPreference, ...] = ()
    excluded_entities: tuple[str, ...] = ()
    overridden_memory_ids: tuple[str, ...] = ()
    clarification: str | None = None


def resolve_recommendation(
    store: JsonlMemoryStore,
    *,
    user_id: str,
    session_id: str,
    beneficiary_id: str | None,
    category: str,
    as_of: datetime,
    requested_entity: str | None = None,
    context_preferences: tuple[ContextPreference, ...] = (),
    change_scope: Literal["request", "unclear"] = "request",
    max_memory_age: timedelta | None = None,
    purpose: Literal["recommendation", "advertising"] = "recommendation",
    use_long_term_memory: bool = True,
) -> RecommendationDecision:
    """只读决策；非本人不使用本人喜好，主体模糊且明确对立时才追问。"""
    use_policy = JsonMemoryUseStore(store.root / "use-policy").get(user_id)
    _check_purpose(use_policy, purpose)
    if type(use_long_term_memory) is not bool:
        raise ValueError("长期记忆开关需要明确布尔值")
    max_memory_age = _memory_age(max_memory_age)
    for value, name in [
        (user_id, "user_id"),
        (session_id, "session_id"),
        (category, "category"),
        (change_scope, "change_scope"),
    ]:
        _text(value, name)
    if len(user_id) > 128 or change_scope not in {"request", "unclear"}:
        raise ValueError("用户或请求范围无效")
    if beneficiary_id is not None:
        _text(beneficiary_id, "beneficiary_id")
    if requested_entity is not None:
        _text(requested_entity, "requested_entity")
        requested_entity = store.canonical_entity(category, requested_entity)
    as_of = _utc(as_of, "as_of")
    for row in context_preferences:
        if (
            not isinstance(row, ContextPreference)
            or row.user_id != user_id
            or row.session_id != session_id
        ):
            raise ValueError("上下文用户或会话不匹配")
    context = tuple(
        replace(row, entity=store.canonical_entity(row.category, row.entity))
        for row in context_preferences
        if row.category == category
        and row.available_at < as_of
        and not (
            row.beneficiary_id in (None, "self")
            and store.evidence_is_deleted(
                user_id, category=row.category, entity=row.entity, available_at=row.available_at
            )
        )
    )
    own = (
        store.active_memories(
            user_id, category=category, as_of=as_of, max_memory_age=max_memory_age
        )
        if use_long_term_memory
        and beneficiary_id in (None, "self")
        and use_policy.allows_personalisation(purpose)
        else ()
    )

    def clarify(question: str) -> RecommendationDecision:
        return RecommendationDecision(
            "clarify", beneficiary_id, requested_entity, clarification=question
        )

    negative = {"exclude", "dislike"}
    conflict = any(row.entity == requested_entity and row.relation in negative for row in own)
    if beneficiary_id is None:
        # 不同品牌不等于冲突；只判断同一实体上的明确对立，不编造距离阈值。
        for assertion in context:
            if assertion.beneficiary_id is not None:
                continue
            conflict |= any(
                row.entity == assertion.entity
                and (
                    (row.relation in negative and assertion.relation == "prefer")
                    or (row.relation == "prefer" and assertion.relation in negative)
                )
                for row in own
            )
        if conflict:
            return clarify("这次是在说你自己的喜好，还是在给其他人推荐？")
        return RecommendationDecision("ready", None, requested_entity)
    grouped: dict[str, list[ContextPreference]] = {}
    for row in context:
        if row.beneficiary_id == beneficiary_id:
            grouped.setdefault(row.entity, []).append(row)
    selected = {}
    for entity, rows in grouped.items():
        latest = max(row.available_at for row in rows)
        current = [row for row in rows if row.available_at == latest]
        if len({row.relation for row in current}) > 1:
            return clarify("当前对话对同一品牌存在相反表达，请确认本次采用哪一种？")
        selected[entity] = min(current, key=lambda row: (row.evidence_ref, row.evidence_text))
    # 本人改主意也可能来自上下文表达，而没有单独指定请求品牌。
    context_conflict = any(
        row.entity in selected
        and (
            (row.relation in negative and selected[row.entity].relation == "prefer")
            or (row.relation == "prefer" and selected[row.entity].relation in negative)
        )
        for row in own
    )
    if change_scope == "unclear" and (conflict or context_conflict):
        return clarify("这次表达只适用于本次推荐，还是要长期修改之前的偏好？")
    if requested_entity in selected and selected[requested_entity].relation in negative:
        return clarify("本次对话同时提出排除和查看该品牌，请确认当前要求。")

    applied, overridden = [], []
    for row in own:
        if row.entity in selected or (row.entity == requested_entity and row.relation in negative):
            overridden.append(row.memory_id)
        else:
            applied.append(row)
    context_used = tuple(selected[entity] for entity in sorted(selected))
    excluded = {row.entity for row in (*applied, *context_used) if row.relation == "exclude"}
    return RecommendationDecision(
        "ready",
        beneficiary_id,
        requested_entity,
        tuple(applied),
        context_used,
        tuple(sorted(excluded)),
        tuple(sorted(overridden)),
    )
