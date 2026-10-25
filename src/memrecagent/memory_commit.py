"""本机宿主持有的提案票据；结构化确认与提交分开，不解释任意自然语言。"""

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta
from typing import Literal
from uuid import uuid4

from memrecagent.attribute_candidates import (
    AttributeChange,
    ContextAttributePreference,
    compile_attribute_changes,
)
from memrecagent.memory import PreferenceMemory, _memory_age, _text, _utc
from memrecagent.memory_evidence_policy import validate_candidate_retention
from memrecagent.memory_extraction import (
    CandidateAssertion,
    EvidenceSpan,
    ExtractionCandidate,
    Subject,
    Turn,
    decode_candidate,
)
from memrecagent.memory_policy import ContextPreference, resolve_recommendation
from memrecagent.memory_proposals import ProposalResult, SessionPerson, propose
from memrecagent.memory_store import JsonlMemoryStore
from memrecagent.memory_use_policy import JsonMemoryUseStore, _check_purpose


class CommitError(ValueError):
    """只提供稳定错误码，不回显原话或内部状态。"""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__("记忆提交失败：" + code)


@dataclass(frozen=True)
class CommitPreview:
    """调用方可以展示的只读预览；提交仅接受宿主持有的票据ID。"""

    ticket_id: str
    action: Literal["apply", "current", "clarify", "noop"]
    proposal: ProposalResult = field(repr=False)


@dataclass(frozen=True)
class CommitOutcome:
    status: Literal["applied", "already_applied", "not_applied"]
    memory_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Ticket:
    preview: CommitPreview
    candidate: ExtractionCandidate
    turns: tuple[Turn, ...]
    created_at: datetime
    snapshot: str
    policy_revision: str
    applied: bool = False
    attribute_changes: tuple[AttributeChange, ...] = ()


class MemoryCommitSession:
    """单进程串行宿主；持有当前票据和已接受的他人会话偏好，不将其写盘。"""

    def __init__(
        self,
        store: JsonlMemoryStore,
        *,
        user_id: str,
        session_id: str,
        max_memory_age: timedelta | None = None,
        purpose: Literal["recommendation", "advertising"] = "recommendation",
    ) -> None:
        _text(user_id, "user_id")
        _text(session_id, "session_id")
        if not isinstance(store, JsonlMemoryStore) or len(user_id) > 128:
            raise CommitError("input")
        self._store = store
        self._policy_store = JsonMemoryUseStore(store.root / "use-policy")
        self._use_policy, self._policy_revision = self._policy_store.snapshot(user_id)
        self._purpose = purpose
        try:
            _check_purpose(self._use_policy, purpose)
        except ValueError:
            raise CommitError("purpose") from None
        self._root = store.root.resolve()
        self._user_id = user_id
        self._session_id = session_id
        self._max_memory_age = _memory_age(max_memory_age)
        self._ticket: _Ticket | None = None
        self._session_preferences: dict[tuple[str, str, str], ContextPreference] = {}
        self._session_attributes: dict[tuple, ContextAttributePreference] = {}

    def _binding(self) -> None:
        self._refresh_policy()
        if self._store.root.resolve() != self._root:
            self._ticket = None
            self._session_preferences.clear()
            self._session_attributes.clear()
            raise CommitError("binding")

    def _refresh_policy(self) -> None:
        policy, revision = self._policy_store.snapshot(self._user_id)
        if revision != self._policy_revision:
            self._session_preferences.clear()
            self._session_attributes.clear()
        self._use_policy, self._policy_revision = policy, revision
        try:
            _check_purpose(policy, self._purpose)
        except ValueError:
            self._ticket = None
            raise CommitError("purpose") from None

    def _accept_context(self, proposal: ProposalResult) -> None:
        """仅在成功接受整轮后发布稳定他人偏好；本次约束不延长为会话偏好。"""
        for row in proposal.session_preferences:
            if row.retention != "session" or row.beneficiary_id in (None, "self"):
                continue
            key = (row.beneficiary_id, row.category, row.entity)
            old = self._session_preferences.get(key)
            if old is None or row.available_at >= old.available_at:
                self._session_preferences[key] = row
        for row in proposal.session_attributes:
            if row.beneficiary_id == "self" or row.retention != "session":
                continue
            key = (row.beneficiary_id, row.category, row.attribute, row.operator, row.is_hard)
            old = self._session_attributes.get(key)
            if old is None or row.available_at >= old.available_at:
                if row.active:
                    self._session_attributes[key] = row
                else:
                    self._session_attributes.pop(key, None)

    def session_attributes(self) -> tuple[ContextAttributePreference, ...]:
        """仅返回已接受的会话属性，调用者仍须按当前人物/条件选择。"""
        self._binding()
        return tuple(self._session_attributes.values())

    def _get(self, ticket_id: str, as_of: datetime) -> _Ticket:
        self._binding()
        ticket = self._ticket
        if ticket is None or ticket.preview.ticket_id != ticket_id:
            raise CommitError("ticket")
        if as_of < ticket.created_at:
            raise CommitError("time")
        return ticket

    def _fresh(self, ticket: _Ticket) -> None:
        if (
            self._policy_revision != ticket.policy_revision
            or self._store.snapshot_token(self._user_id) != ticket.snapshot
        ):
            self._ticket = None
            raise CommitError("stale")

    def prepare(
        self,
        candidate: ExtractionCandidate,
        *,
        turns: tuple[Turn, ...],
        as_of: datetime,
        people: tuple[SessionPerson, ...] = (),
        attribute_changes: tuple[AttributeChange, ...] = (),
    ) -> CommitPreview:
        """核验证据并冻结候选，不写盘；新的有效预览取代上一张票据。"""
        return self._prepare(
            candidate, turns=turns, as_of=as_of, people=people, attribute_changes=attribute_changes
        )

    def _prepare(
        self,
        candidate: ExtractionCandidate,
        *,
        turns: tuple[Turn, ...],
        as_of: datetime,
        people: tuple[SessionPerson, ...] = (),
        question_context: Turn | None = None,
        attribute_changes: tuple[AttributeChange, ...] = (),
    ) -> CommitPreview:
        self._binding()
        as_of = _utc(as_of, "as_of")
        if not isinstance(candidate, ExtractionCandidate):
            raise CommitError("input")
        candidate = decode_candidate(json.dumps(asdict(candidate), ensure_ascii=False), turns=turns)
        known = {t.turn_id: t for t in turns}
        snapshot = self._store.snapshot_token(self._user_id)
        proposal = propose(
            candidate,
            store=self._store,
            user_id=self._user_id,
            session_id=self._session_id,
            turns=turns,
            people=people,
            as_of=as_of,
            max_memory_age=self._max_memory_age,
            purpose=self._purpose,
        )
        attributes, context_attributes, unknown_attributes = compile_attribute_changes(
            attribute_changes,
            store=self._store,
            user_id=self._user_id,
            session_id=self._session_id,
            turns=turns,
            as_of=as_of,
        )
        validate_candidate_retention(candidate.assertions, attribute_changes, turns)
        proposal = replace(
            proposal, proposed_attributes=attributes, session_attributes=context_attributes
        )
        if unknown_attributes:
            proposal = replace(
                proposal,
                proposed_memories=(),
                proposed_attributes=(),
                session_attributes=(),
                decision=replace(
                    proposal.decision,
                    status="clarify",
                    clarification="请完整重述这项属性要求，明确是谁、具体条件，以及只对本次还是长期有效。",
                ),
            )
        if any(u.field in ("entity", "relation") for u in candidate.unresolved):
            # 即使未解片段与明确断言不重叠，也不能先提交半轮结果。
            proposal = replace(
                proposal,
                proposed_memories=(),
                decision=replace(
                    proposal.decision,
                    status="clarify",
                    clarification="还有未明确的品牌或态度，请完整重述本次偏好。",
                ),
            )
        if proposal.decision.status == "ready" and any(
            a.subject.kind == "self"
            and a.relation != "none"
            and a.scope == "unclear"
            and not any(
                self._store.evidence_is_deleted(
                    self._user_id,
                    category=a.category,
                    entity=a.entity,
                    occurred_at=known[s.turn_id].occurred_at,
                    available_at=known[s.turn_id].available_at,
                )
                for s in a.evidence + a.subject.references
            )
            for a in candidate.assertions
        ):
            proposal = replace(
                proposal,
                proposed_memories=(),
                decision=replace(
                    proposal.decision,
                    status="clarify",
                    clarification="这次偏好只适用于本次，还是需要长期保存？",
                ),
            )
        # 取消一项不存在的长期态度无需建立空链，也不把它当作新的会话偏好。
        empty_neutral = {
            (m.category, m.entity)
            for m in proposal.proposed_memories
            if m.relation == "neutral" and m.supersedes is None
        }
        proposal = replace(
            proposal,
            proposed_memories=tuple(
                m for m in proposal.proposed_memories if (m.category, m.entity) not in empty_neutral
            ),
            session_preferences=tuple(
                c
                for c in proposal.session_preferences
                if not (c.relation == "neutral" and (c.category, c.entity) in empty_neutral)
            ),
            decision=replace(
                proposal.decision,
                context_preferences=tuple(
                    c
                    for c in proposal.decision.context_preferences
                    if not (c.relation == "neutral" and (c.category, c.entity) in empty_neutral)
                ),
            ),
        )
        request = candidate.request
        if (
            proposal.decision.status == "ready"
            and request.is_recommendation
            and request.category is not None
        ):
            # 明确点名品牌可以临时覆盖旧排除；同轮的新排除仍由策略检查矛盾。
            prior = tuple(
                row
                for row in self._session_preferences.values()
                if not (
                    request.requested_entity is not None
                    and row.entity
                    == self._store.canonical_entity(request.category, request.requested_entity)
                    and row.relation in {"exclude", "dislike"}
                )
            )
            decision = resolve_recommendation(
                self._store,
                user_id=self._user_id,
                session_id=self._session_id,
                beneficiary_id=proposal.decision.beneficiary_id,
                category=request.category,
                as_of=as_of,
                requested_entity=request.requested_entity,
                context_preferences=prior + proposal.session_preferences,
                change_scope="unclear" if request.change_scope == "unclear" else "request",
                max_memory_age=self._max_memory_age,
                purpose=self._purpose,
            )
            proposal = replace(
                proposal,
                decision=decision,
                proposed_memories=proposal.proposed_memories if decision.status == "ready" else (),
            )
        if question_context is not None:
            # 助手问题是短答的语境，不伪装为用户态度证据。
            def contextualize[T: (PreferenceMemory, ContextPreference)](row: T) -> T:
                reference = (
                    f";context:assistant:{question_context.turn_id}"
                    f"[0:{len(question_context.content)}]"
                )
                return replace(
                    row,
                    evidence_ref=row.evidence_ref + reference,
                    evidence_text=row.evidence_text
                    + f"\n[assistant语境，非用户态度 {question_context.turn_id}]\n"
                    + question_context.content,
                )

            proposal = replace(
                proposal,
                proposed_memories=tuple(contextualize(m) for m in proposal.proposed_memories),
                session_preferences=tuple(contextualize(c) for c in proposal.session_preferences),
                decision=replace(
                    proposal.decision,
                    context_preferences=tuple(
                        contextualize(c) for c in proposal.decision.context_preferences
                    ),
                ),
            )
        if not self._use_policy.long_term_updates:
            proposal = replace(proposal, proposed_memories=(), proposed_attributes=())
        if proposal.decision.status == "clarify":
            proposal = replace(
                proposal, proposed_memories=(), proposed_attributes=(), session_attributes=()
            )
        if proposal.decision.status == "clarify":
            action = "clarify"
        elif proposal.proposed_memories or proposal.proposed_attributes:
            action = "apply"
        elif (
            proposal.session_preferences
            or proposal.decision.context_preferences
            or proposal.decision.requested_entity is not None
            or proposal.session_attributes
        ):
            action = "current"
        else:
            action = "noop"
        preview = CommitPreview(uuid4().hex, action, proposal)
        self._ticket = _Ticket(
            preview,
            candidate,
            tuple(turns),
            as_of,
            snapshot,
            self._policy_revision,
            attribute_changes=attribute_changes,
        )
        return preview

    def commit(self, ticket_id: str, *, as_of: datetime) -> CommitOutcome:
        """只提交当前宿主票据；澄清、过期、归属变化或半批失败均不能写入。"""
        as_of = _utc(as_of, "as_of")
        ticket = self._get(ticket_id, as_of)
        memories = (
            ticket.preview.proposal.proposed_memories + ticket.preview.proposal.proposed_attributes
        )
        if ticket.policy_revision != self._policy_revision:
            self._ticket = None
            raise CommitError("stale")
        if ticket.applied:
            current = {
                m.memory_id: m
                for m in (
                    *self._store.list_memories(self._user_id, as_of=as_of),
                    *self._store.list_attribute_memories(self._user_id, as_of=as_of),
                )
            }
            if not all(current.get(m.memory_id) == m for m in memories):
                self._ticket = None
                raise CommitError("stale")
            return CommitOutcome("already_applied", tuple(m.memory_id for m in memories))
        self._fresh(ticket)
        if ticket.preview.action == "clarify":
            raise CommitError("confirmation_required")
        if ticket.preview.action != "apply":
            self._accept_context(ticket.preview.proposal)
            return CommitOutcome("not_applied")
        try:
            self._store.add_many(memories, expected_token=ticket.snapshot)
        except (ValueError, OSError):
            raise CommitError("storage") from None
        self._ticket = replace(ticket, applied=True)
        self._accept_context(ticket.preview.proposal)
        return CommitOutcome("applied", tuple(m.memory_id for m in memories))

    def confirm(
        self,
        ticket_id: str,
        *,
        reply: Turn,
        subject: Literal["self", "other"],
        scope: Literal["current", "long_term"],
        as_of: datetime,
        person_reference: EvidenceSpan | None = None,
        question: Turn | None = None,
    ) -> CommitPreview:
        """消费可信调用方解析的明确选择；保留实际用户答复，不自动写盘。"""
        as_of = _utc(as_of, "as_of")
        ticket = self._get(ticket_id, as_of)
        self._fresh(ticket)
        if ticket.preview.action != "clarify":
            raise CommitError("not_pending")
        if ticket.attribute_changes:
            # 属性未解要求完整重述；品牌旧短答协议不能丢掉类型化条件后提交半轮。
            raise CommitError("clarification_target")
        if (
            not isinstance(reply, Turn)
            or reply.role != "user"
            or reply.turn_id in {t.turn_id for t in ticket.turns}
            or reply.occurred_at <= ticket.created_at
            or reply.occurred_at >= as_of
            or reply.available_at >= as_of
        ):
            raise CommitError("reply")
        if question is not None and (
            not isinstance(question, Turn)
            or question.role != "assistant"
            or question.turn_id in {t.turn_id for t in ticket.turns} | {reply.turn_id}
            or question.content != ticket.preview.proposal.decision.clarification
            or question.occurred_at < ticket.created_at
            or question.available_at >= reply.occurred_at
            or question.available_at >= as_of
        ):
            raise CommitError("question")
        if subject not in ("self", "other") or scope not in ("current", "long_term"):
            raise CommitError("choice")
        if subject == "other" and (
            scope != "current"
            or not isinstance(person_reference, EvidenceSpan)
            or person_reference.turn_id != reply.turn_id
        ):
            raise CommitError("choice")
        if subject == "self" and person_reference is not None:
            raise CommitError("choice")
        active = tuple(a for a in ticket.candidate.assertions if a.relation != "none")
        targets = {
            (a.category, self._store.canonical_entity(a.category, a.entity), a.relation)
            for a in active
        }
        subjects = {
            (a.subject.kind, a.subject.references if a.subject.kind == "other" else ())
            for a in active
        }
        request = ticket.candidate.request
        if (
            request.is_recommendation
            and request.beneficiary.kind == "unknown"
            and any(u.field == "subject" for u in ticket.candidate.unresolved)
            and any(a.subject.kind != "unknown" for a in active)
        ):
            # 只在问推荐对象时，短答不能把已明确归属的态度改派给该对象。
            # 混合表达需完整重述，不能用一次人物选择同时确认不同命题。
            raise CommitError("clarification_target")
        entities = {
            (a.category, self._store.canonical_entity(a.category, a.entity)) for a in active
        }
        if request.is_recommendation and request.requested_entity is not None:
            category = request.category or (active[0].category if active else None)
            entities.add(
                (
                    category,
                    self._store.canonical_entity(category, request.requested_entity)
                    if category is not None
                    else request.requested_entity,
                )
            )
        if (
            len(targets) > 1
            or any(u.field in ("relation", "entity") for u in ticket.candidate.unresolved)
            or len(subjects) > 1
            or len(entities) > 1
        ):
            raise CommitError("clarification_target")
        if not active and (scope != "current" or not ticket.candidate.request.is_recommendation):
            raise CommitError("clarification_target")
        if not active and request.beneficiary.kind not in ("unknown", subject):
            # 这是改写已明确请求对象，应重新解析完整新请求，不能悄悄忽略答复。
            raise CommitError("clarification_target")
        answer = EvidenceSpan(reply.turn_id, 0, len(reply.content))
        who = Subject(subject, (person_reference,) if subject == "other" else (answer,))
        assertions = ()
        if active:
            category, _, relation = next(iter(targets))
            # 规范键只用于上面的同目标守卫；严格原文复核仍需票据中的别名写法。
            # _prepare通过原证据检查后，propose再按同一可信表规范化保存键。
            entity = active[0].entity
            sources = tuple(dict.fromkeys(s for a in active for s in a.evidence)) + (answer,)
            assertions = (CandidateAssertion(who, category, entity, relation, scope, sources),)
        if request.is_recommendation:
            # 喜好主体与推荐受益人分开；只澄清原本未知的受益人。
            request = replace(
                request,
                beneficiary=who if request.beneficiary.kind == "unknown" else request.beneficiary,
                change_scope="request" if scope == "current" else "long_term",
                evidence=request.evidence + (answer,),
            )
        candidate = ExtractionCandidate(request, assertions, ())
        return self._prepare(
            candidate,
            turns=ticket.turns + ((question,) if question is not None else ()) + (reply,),
            as_of=as_of,
            people=ticket.preview.proposal.people,
            question_context=question,
        )

    def cancel(self, ticket_id: str) -> None:
        """用户取消或会话结束时丢弃当前票据，不能继续用旧确认提交。"""
        self._binding()
        if self._ticket is None or self._ticket.preview.ticket_id != ticket_id:
            raise CommitError("ticket")
        self._ticket = None
