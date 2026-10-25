"""把已验证的语言候选转成只读提案；存储执行权留给可信调用方。"""

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from hashlib import sha256

from memrecagent.attribute_candidates import ContextAttributePreference
from memrecagent.attribute_memory import AttributeMemory
from memrecagent.memory import PreferenceMemory, _memory_age, _text, _utc
from memrecagent.memory_evidence_policy import (
    validate_candidate_retention,
    validate_persistent_evidence,
)
from memrecagent.memory_extraction import (
    EvidenceSpan,
    ExtractionCandidate,
    ExtractionError,
    Subject,
    Turn,
    decode_candidate,
    evidence_text,
)
from memrecagent.memory_policy import (
    ContextPreference,
    RecommendationDecision,
    resolve_recommendation,
)
from memrecagent.memory_store import JsonlMemoryStore
from memrecagent.memory_use_policy import JsonMemoryUseStore, _check_purpose


@dataclass(frozen=True)
class SessionPerson:
    """身份仅在本用户本会话有效，ID由可信来源位置生成。"""

    person_id: str
    references: tuple[EvidenceSpan, ...]


@dataclass(frozen=True)
class ProposalResult:
    decision: RecommendationDecision
    proposed_memories: tuple[PreferenceMemory, ...]
    session_preferences: tuple[ContextPreference, ...]
    people: tuple[SessionPerson, ...]
    proposed_attributes: tuple[AttributeMemory, ...] = ()
    session_attributes: tuple[ContextAttributePreference, ...] = ()


def _digest(value: object) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def session_person_id(
    user_id: str, session_id: str, references: tuple[EvidenceSpan, ...], turns: tuple[Turn, ...]
) -> str:
    """品牌和属性共用原始人物锚点规则，身份不由模型自由命名。"""
    if not references:
        raise ExtractionError("person_anchor")
    anchor = references[0]
    turn = next((t for t in turns if t.turn_id == anchor.turn_id), None)
    if (
        turn is None
        or turn.role != "user"
        or type(anchor.start) is not int
        or type(anchor.end) is not int
        or not 0 <= anchor.start < anchor.end <= len(turn.content)
    ):
        raise ExtractionError("person_anchor")
    return "person:" + _digest([user_id, session_id, asdict(anchor)])


def propose(
    candidate: ExtractionCandidate,
    *,
    store: JsonlMemoryStore,
    user_id: str,
    session_id: str,
    turns: tuple[Turn, ...],
    people: tuple[SessionPerson, ...],
    as_of: datetime,
    max_memory_age: timedelta | None = None,
    purpose: str = "recommendation",
) -> ProposalResult:
    """只使用当时可用的证据；任一待澄清冲突阻止本次全部长期提案。"""
    use_policy = JsonMemoryUseStore(store.root / "use-policy").get(user_id)
    _check_purpose(use_policy, purpose)
    max_memory_age = _memory_age(max_memory_age)
    _text(user_id, "user_id")
    _text(session_id, "session_id")
    if len(user_id) > 128 or not isinstance(candidate, ExtractionCandidate):
        raise ExtractionError("input")
    as_of = _utc(as_of, "as_of")
    # 再走同一严格边界，防止手工构造dataclass绕过解码校验。
    candidate = decode_candidate(json.dumps(asdict(candidate), ensure_ascii=False), turns=turns)
    validate_candidate_retention(candidate.assertions, (), turns)
    for unresolved in candidate.unresolved:
        if unresolved.field not in {"subject", "relation", "entity"}:
            continue
        for assertion in candidate.assertions:
            if assertion.relation == "none" or (
                unresolved.field == "subject" and assertion.subject.kind != "self"
            ):
                continue
            if any(
                a.turn_id == b.turn_id and max(a.start, b.start) < min(a.end, b.end)
                for a in assertion.evidence
                for b in unresolved.evidence
            ):
                raise ExtractionError("inconsistent_" + unresolved.field)
    if any(t.occurred_at >= as_of or t.available_at >= as_of for t in turns):
        raise ExtractionError("future_evidence")
    known = {t.turn_id: t for t in turns}
    order = {t.turn_id: index for index, t in enumerate(turns)}

    def canonical(spans: tuple[EvidenceSpan, ...]) -> tuple[EvidenceSpan, ...]:
        return tuple(sorted(set(spans), key=lambda s: (order[s.turn_id], s.start, s.end)))

    def person_id(references: tuple[EvidenceSpan, ...]) -> str:
        return session_person_id(user_id, session_id, references, turns)

    registry = {}
    for person in people:
        if not isinstance(person, SessionPerson) or person.person_id != person_id(
            person.references
        ):
            raise ExtractionError("person_session")
        if person.person_id in registry:
            raise ExtractionError("duplicate_person")
        registry[person.person_id] = person

    def identity(subject: Subject) -> str | None:
        if subject.kind == "self":
            return "self"
        if subject.kind == "unknown":
            return None
        identity_id = person_id(subject.references)
        registry.setdefault(identity_id, SessionPerson(identity_id, subject.references))
        return identity_id

    request = candidate.request
    if not request.is_recommendation:
        # 无请求的占位字段不能压过断言已明确的长期范围或人物。
        request = replace(
            request,
            beneficiary=Subject("unknown", ()),
            category=None,
            requested_entity=None,
            change_scope="request",
            evidence=(),
        )
    beneficiary = identity(request.beneficiary)
    if (
        request.is_recommendation
        and beneficiary in (None, "self")
        and request.category is not None
        and request.requested_entity is not None
        and any(
            store.evidence_is_deleted(
                user_id,
                category=request.category,
                entity=request.requested_entity,
                occurred_at=known[s.turn_id].occurred_at,
                available_at=known[s.turn_id].available_at,
            )
            for s in canonical(request.evidence + request.beneficiary.references)
        )
    ):
        raise ExtractionError("deleted_request")
    active_assertions = [a for a in candidate.assertions if a.relation != "none"]
    merged = {}
    for assertion in active_assertions:
        if assertion.subject.kind == "self":
            # 必须在跨轮合并前逐证据判断；max时间会让新表达夹带已删原文。
            retired = [
                store.evidence_is_deleted(
                    user_id,
                    category=assertion.category,
                    entity=store.canonical_entity(assertion.category, assertion.entity),
                    occurred_at=known[s.turn_id].occurred_at,
                    available_at=known[s.turn_id].available_at,
                )
                for s in canonical(assertion.evidence + assertion.subject.references)
            ]
            if any(retired):
                if not all(retired):
                    # 单条断言混合新旧证据时不擅自裁剪后重新解释模型语义。
                    raise ExtractionError("deleted_evidence")
                continue
        key = (
            identity(assertion.subject),
            assertion.category,
            store.canonical_entity(assertion.category, assertion.entity),
            assertion.relation,
            assertion.scope,
        )
        previous = merged.get(key)
        if previous is not None:
            assertion = replace(
                previous,
                evidence=canonical(previous.evidence + assertion.evidence),
                subject=replace(
                    previous.subject,
                    references=canonical(
                        previous.subject.references + assertion.subject.references
                    ),
                ),
            )
        merged[key] = assertion
    active_assertions = list(merged.values())
    categories = {a.category for a in active_assertions}
    category = request.category or (next(iter(categories)) if len(categories) == 1 else None)
    # 没有推荐请求时仍可判断当前表达是否与本人历史冲突。
    if not request.is_recommendation:
        subjects = {a.subject.kind for a in active_assertions}
        beneficiary = "self" if subjects == {"self"} else None
    contexts, proposals = [], []
    uncertainty = False
    seen_relations: dict[tuple[str | None, str, str], set[str]] = {}
    seen_scopes: dict[tuple[str | None, str, str], set[str]] = {}
    history = tuple(
        m for cat in categories for m in store.list_memories(user_id, category=cat, as_of=as_of)
    )
    recorded = {m.memory_id for m in history}
    active_by_key = {
        (m.category, m.entity): m
        for m in history
        if m.memory_id not in {row.supersedes for row in history if row.supersedes is not None}
    }
    for assertion in active_assertions:
        who = identity(assertion.subject)
        evidence = canonical(assertion.evidence)
        sources = canonical(evidence + assertion.subject.references)
        occurred_at = max(known[s.turn_id].occurred_at for s in evidence)
        if any(known[s.turn_id].occurred_at > occurred_at for s in assertion.subject.references):
            raise ExtractionError("future_antecedent")
        available_at = max(known[s.turn_id].available_at for s in sources)
        reference = session_id + ":" + ",".join(f"{s.turn_id}[{s.start}:{s.end}]" for s in sources)
        original = evidence_text(sources, turns)
        if who == "self" and assertion.scope == "long_term":
            validate_persistent_evidence(sources, turns)
        memory_id = "memory:" + _digest(
            [
                user_id,
                session_id,
                assertion.category,
                store.canonical_entity(assertion.category, assertion.entity),
                assertion.relation,
                [asdict(s) for s in sources],
            ]
        )
        if who == "self" and memory_id in recorded:
            # 已应用的旧证据不再作为新上下文；有效状态由存储修订链决定。
            continue
        contexts.append(
            ContextPreference(
                user_id,
                session_id,
                who,
                assertion.category,
                store.canonical_entity(assertion.category, assertion.entity),
                assertion.relation,
                reference,
                original,
                available_at,
                retention=(
                    "session"
                    if who not in (None, "self") and assertion.scope == "long_term"
                    else "request"
                ),
            )
        )
        key = (
            who,
            assertion.category,
            store.canonical_entity(assertion.category, assertion.entity),
        )
        seen_relations.setdefault(key, set()).add(assertion.relation)
        seen_scopes.setdefault(key, set()).add(assertion.scope)
        old = active_by_key.get(
            (assertion.category, store.canonical_entity(assertion.category, assertion.entity))
        )
        if (
            who == "self"
            and assertion.scope == "unclear"
            and old
            and old.relation != assertion.relation
        ):
            uncertainty = True
        if who != "self" or assertion.scope != "long_term":
            continue
        if old and (occurred_at <= old.occurred_at or available_at < old.available_at):
            raise ExtractionError("revision_time")
        proposals.append(
            PreferenceMemory(
                memory_id,
                user_id,
                assertion.category,
                store.canonical_entity(assertion.category, assertion.entity),
                assertion.relation,
                reference,
                original,
                occurred_at,
                available_at,
                supersedes=old.memory_id if old else None,
            )
        )

    def result(decision: RecommendationDecision) -> ProposalResult:
        return ProposalResult(
            decision,
            tuple(proposals) if decision.status == "ready" and use_policy.long_term_updates else (),
            tuple(contexts),
            tuple(registry.values()),
        )

    if any(u.field == "subject" for u in candidate.unresolved):
        # 显式未解人物不能降级为普通泛化请求，也不能先发布同轮的明确偏好。
        return result(
            RecommendationDecision(
                "clarify",
                beneficiary,
                request.requested_entity,
                clarification="这次是给谁推荐？请明确人物对象。"
                if request.is_recommendation and request.beneficiary.kind == "unknown"
                else "这条喜好属于谁？请明确人物对象。",
            )
        )
    if any(len(values) > 1 for values in (*seen_relations.values(), *seen_scopes.values())):
        return result(
            RecommendationDecision(
                "clarify",
                beneficiary,
                request.requested_entity,
                clarification="同一对象的当前表达存在冲突，请确认采用哪种态度。",
            )
        )
    if uncertainty or any(u.field == "scope" for u in candidate.unresolved):
        return result(
            RecommendationDecision(
                "clarify",
                beneficiary,
                request.requested_entity,
                clarification="这次表达只适用于本次推荐，还是要长期修改之前的偏好？",
            )
        )
    # 推荐对象明确不代表每条态度的主语也明确，单独检查未知态度的冲突。
    for cat in {c.category for c in contexts if c.beneficiary_id is None}:
        subject_check = resolve_recommendation(
            store,
            user_id=user_id,
            session_id=session_id,
            beneficiary_id=None,
            category=cat,
            as_of=as_of,
            context_preferences=tuple(contexts),
            max_memory_age=max_memory_age,
            purpose=purpose,
        )
        if subject_check.status == "clarify":
            return result(subject_check)
    if category is None:
        if len(categories) > 1:
            return result(
                RecommendationDecision(
                    "clarify",
                    beneficiary,
                    request.requested_entity,
                    clarification="请确认本次商品类别。",
                )
            )
        return result(RecommendationDecision("ready", beneficiary, request.requested_entity))
    scope = "unclear" if request.change_scope == "unclear" else "request"
    return result(
        resolve_recommendation(
            store,
            user_id=user_id,
            session_id=session_id,
            beneficiary_id=beneficiary,
            category=category,
            as_of=as_of,
            requested_entity=request.requested_entity,
            context_preferences=tuple(contexts),
            change_scope=scope,
            max_memory_age=max_memory_age,
            purpose=purpose,
        )
    )
