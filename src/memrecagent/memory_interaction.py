"""接收实际双向历史的记忆Hook；分析、预览和发布各自有明确边界。"""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Literal
from uuid import uuid4

from memrecagent.attribute_candidates import ContextAttributePreference
from memrecagent.deepseek_client import ModelReply
from memrecagent.memory import PreferenceMemory, _text, _utc
from memrecagent.memory_analyst import (
    AnalysisCall,
    AnalystError,
    AnalystTask,
    MemoryAnalyst,
    MemoryView,
    build_memory_view,
    inspect_draft,
)
from memrecagent.memory_commit import CommitError, CommitOutcome, CommitPreview, MemoryCommitSession
from memrecagent.memory_extraction import Subject, Turn, validate_turns
from memrecagent.memory_interaction_extraction import (
    Intent,
    InteractionCandidate,
    PreferenceClue,
    build_interaction_messages,
    decode_interaction,
)
from memrecagent.memory_proposals import SessionPerson, session_person_id
from memrecagent.memory_store import JsonlMemoryStore
from memrecagent.memory_use_policy import JsonMemoryUseStore, MemoryUsePolicy, _check_purpose
from memrecagent.request_attributes import AttributeRequestCandidate

ResponseAction = Literal[
    "continue", "answer", "compare", "recommend", "clarify", "feedback_target_required"
]
Complete = Callable[[list[dict[str, str]]], ModelReply]
_SCOPE_QUESTIONS = {
    "这次偏好只适用于本次，还是需要长期保存？",
    "这次表达只适用于本次推荐，还是要长期修改之前的偏好？",
}
_AMBIGUOUS = {"嗯", "好", "可以", "行", "对", "是", "随便", "ok", "yes"}


@dataclass(frozen=True)
class InteractionPreview:
    """调用方可据此回答、追问或展示；不包含商品排名。"""

    ticket_id: str
    intents: tuple[Intent, ...]
    topic_entities: tuple[str, ...]
    clues: tuple[PreferenceClue, ...] = field(repr=False)
    context_memories: tuple[PreferenceMemory, ...] = field(repr=False)
    memory_preview: CommitPreview = field(repr=False)
    response_action: ResponseAction
    analysis_calls: tuple[AnalysisCall, ...] = ()
    attribute_request: AttributeRequestCandidate | None = None


@dataclass(frozen=True)
class _InteractionTicket:
    preview: InteractionPreview
    analysis: InteractionCandidate
    created_at: datetime
    accepted: bool = False
    is_recommendation: bool = False


class ConversationMemorySession:
    """串行对话工作区；唯一发布点为commit，其他人的偏好仅留在会话内。"""

    def __init__(
        self,
        store: JsonlMemoryStore,
        *,
        user_id: str,
        session_id: str,
        max_memory_age: timedelta | None = None,
        purpose: Literal["recommendation", "advertising"] = "recommendation",
        category: str = "phone",
    ) -> None:
        self._category = _text(category, "category")
        if len(category) > 128:
            raise CommitError("category")
        self._store = store
        self._user_id = user_id
        self._session_id = session_id
        self._max_age = max_memory_age
        self._purpose = purpose
        self._policy_store = JsonMemoryUseStore(store.root / "use-policy")
        self._use_policy, self._policy_revision = self._policy_store.snapshot(user_id)
        self._host = self._new_host()
        self._root = store.root.resolve()
        self._snapshot = store.snapshot_token(user_id)
        self._turns: tuple[Turn, ...] = ()
        self._clues: tuple[PreferenceClue, ...] = ()
        self._people: tuple[SessionPerson, ...] = ()
        self._current: _InteractionTicket | None = None
        self._pending: CommitPreview | None = None
        self._asked: Turn | None = None
        self._needs_reset = False
        self._analysis_calls: tuple[AnalysisCall, ...] = ()

    def _new_host(self) -> MemoryCommitSession:
        return MemoryCommitSession(
            self._store,
            user_id=self._user_id,
            session_id=self._session_id,
            max_memory_age=self._max_age,
            purpose=self._purpose,
        )

    def get_use_policy(self) -> MemoryUsePolicy:
        """读取本用户设置；不从用户态度猜广告许可。"""
        return self._policy_store.get(self._user_id)

    def set_use_policy(self, policy: MemoryUsePolicy) -> None:
        """可信宿主显式更新，成功后废弃旧原话、工作线索和提案。"""
        if self._store.root.resolve() != self._root:
            raise CommitError("binding")
        self._policy_store.set(self._user_id, policy)
        self._use_policy, self._policy_revision = self._policy_store.snapshot(self._user_id)
        self._turns, self._clues, self._people = (), (), ()
        self._invalidate()

    @property
    def turns(self) -> tuple[Turn, ...]:
        """实际发生的前缀；模型失败也不伪装成该轮未发生。"""
        return self._turns

    @property
    def working_clues(self) -> tuple[PreferenceClue, ...]:
        """已接受但未持久化的线索；新实例不继承。"""
        return self._clues

    @property
    def last_analysis_calls(self) -> tuple[AnalysisCall, ...]:
        """最近一次prepare的双阶段实际尝试，失败或过期仍可核对成本。"""
        return self._analysis_calls

    def _invalidate(self) -> None:
        self._current = None
        self._pending = None
        self._asked = None
        try:
            self._host = self._new_host()
        except CommitError as error:
            if error.code != "purpose":
                raise
            # 撤销广告许可已成功，不能为了重建一个禁用宿主而把设置操作报失败。
            # 所有使用入口先fresh核验purpose，重新授权后才重新创建内部宿主。
            self._host = None

    def clear_context(self) -> None:
        """释放本实例原话/人物/票据；不写磁盘，下次fresh才安全重建内部宿主。"""
        self._turns, self._clues, self._people = (), (), ()
        self._current, self._pending, self._asked = None, None, None
        self._host = None
        self._needs_reset = True

    def _fresh(self, expected_policy_revision: str | None = None) -> None:
        if self._store.root.resolve() != self._root:
            self._turns = ()
            self._clues = ()
            self._people = ()
            self._current = None
            self._pending = None
            self._asked = None
            self._needs_reset = True
            raise CommitError("binding")
        policy, revision = self._policy_store.snapshot(self._user_id)
        if revision != self._policy_revision or (
            expected_policy_revision is not None and revision != expected_policy_revision
        ):
            self._use_policy, self._policy_revision = policy, revision
            self._turns, self._clues, self._people = (), (), ()
            self._invalidate()
            raise CommitError("stale")
        try:
            _check_purpose(policy, self._purpose)
        except ValueError:
            raise CommitError("purpose") from None
        if self._needs_reset:
            self._invalidate()
            self._needs_reset = False
        snapshot = self._store.snapshot_token(self._user_id)
        if snapshot != self._snapshot:
            # 外部删除/纠正后不再把旧原话发给模型；显式要求新的实际前缀。
            self._turns = ()
            self._clues = ()
            self._people = ()
            self._invalidate()
            self._snapshot = snapshot
            raise CommitError("stale")

    def _appendable(self, turn: Turn, role: str) -> tuple[Turn, ...]:
        if not isinstance(turn, Turn) or turn.role != role:
            raise CommitError("turn")
        turns = self._turns + (turn,)
        validate_turns(turns)
        return turns

    def observe_assistant(self, turn: Turn) -> None:
        """只登记已经实际输出的助手消息；不调用模型、不持久化。"""
        self._fresh()
        turns = self._appendable(turn, "assistant")
        self._turns = turns
        self._asked = (
            turn
            if self._pending is not None
            and turn.content == self._pending.proposal.decision.clarification
            and self._current is not None
            and turn.occurred_at >= self._current.created_at
            else None
        )

    def _drop_preview(self) -> None:
        if self._current is not None:
            try:
                self._host.cancel(self._current.preview.memory_preview.ticket_id)
            except CommitError:
                pass
        self._current = None
        self._pending = None
        self._asked = None

    def prepare_user(
        self,
        turn: Turn,
        *,
        complete: Complete,
        as_of: datetime,
        analysis_mode: Literal["direct", "analyst", "analyst_blind"] = "direct",
    ) -> InteractionPreview:
        """默认单次分析；显式双阶段仅用最终候选，失败撤销旧提交入口。"""
        if analysis_mode not in ("direct", "analyst", "analyst_blind"):
            raise CommitError("analysis_mode")
        self._analysis_calls = ()
        self._fresh()
        as_of = _utc(as_of, "as_of")
        turns = self._appendable(turn, "user")
        if any(t.occurred_at >= as_of or t.available_at >= as_of for t in turns):
            raise CommitError("time")
        asked = self._asked if self._turns and self._turns[-1] == self._asked else None
        self._turns = turns
        policy_revision = self._policy_revision
        try:
            # 先识别意图再选相关背景；无关聊天的分析不携带全部持久化偏好。
            if analysis_mode != "direct":
                analysis = self._analyze_with_role(
                    turns,
                    asked,
                    complete,
                    as_of,
                    policy_revision,
                    expose_draft=analysis_mode == "analyst",
                )
            else:
                reply = complete(
                    build_interaction_messages(
                        turns,
                        (),
                        pending_question=asked,
                        known_subjects=self._known_subjects(),
                        category=self._category,
                    )
                )
                if not isinstance(reply, ModelReply):
                    raise CommitError("model_reply")
                analysis = decode_interaction(reply.content, turns=turns)
            self._fresh(policy_revision)
            self._validate_category(analysis)
            self._validate_clues(analysis.clues)
            is_recommendation = analysis.candidate.request.is_recommendation
            if analysis.confirmation is not None:
                # 确认信封本身为空操作，原请求类型由宿主票据接续，不要求模型重提。
                is_recommendation = (
                    analysis.confirmation.choice.outcome != "cancel"
                    and self._current is not None
                    and self._current.is_recommendation
                )
                preview = self._confirm(analysis, turn, asked, as_of)
            else:
                people = self._people
                attribute = analysis.attribute_request
                if attribute is not None and attribute.subject.kind == "other":
                    person = SessionPerson(
                        session_person_id(
                            self._user_id, self._session_id, attribute.subject.references, turns
                        ),
                        attribute.subject.references,
                    )
                    if not any(p.person_id == person.person_id for p in people):
                        people += (person,)
                for change in analysis.attribute_changes:
                    if change.subject.kind == "other":
                        person = SessionPerson(
                            session_person_id(
                                self._user_id, self._session_id, change.subject.references, turns
                            ),
                            change.subject.references,
                        )
                        if not any(p.person_id == person.person_id for p in people):
                            people += (person,)
                preview = self._host.prepare(
                    analysis.candidate,
                    turns=turns,
                    as_of=as_of,
                    people=people,
                    attribute_changes=analysis.attribute_changes,
                )
            context = self._background(
                analysis, preview, as_of, is_recommendation=is_recommendation
            )
            action = self._action(analysis, preview, is_recommendation=is_recommendation)
            attribute_request = analysis.attribute_request
            if analysis.confirmation is not None and self._current is not None:
                pending_attribute = self._current.preview.attribute_request
                if pending_attribute is not None:
                    choice = analysis.confirmation.choice
                    if choice.outcome == "unclear":
                        attribute_request = pending_attribute
                    elif choice.outcome == "confirm":
                        who = pending_attribute.subject
                        if who.kind == "unknown":
                            who = Subject(
                                choice.subject,
                                (choice.person_reference,) if choice.subject == "other" else (),
                            )
                        attribute_request = replace(
                            pending_attribute,
                            subject=who,
                            evidence=pending_attribute.evidence + choice.evidence,
                            conditions=tuple(
                                replace(c, evidence=c.evidence + choice.evidence)
                                for c in pending_attribute.conditions
                            ),
                        )
            result = InteractionPreview(
                uuid4().hex,
                analysis.intents,
                analysis.topic_entities,
                analysis.clues,
                context,
                preview,
                action,
                self._analysis_calls,
                attribute_request,
            )
            self._current = _InteractionTicket(
                result, analysis, as_of, is_recommendation=is_recommendation
            )
            self._pending = preview if preview.action == "clarify" else None
            self._asked = None
            return result
        except Exception:
            self._drop_preview()
            raise

    def _validate_category(self, analysis: InteractionCandidate) -> None:
        """品类由可信宿主绑定，模型不能跨域提交或套用手机属性。"""
        categories = [analysis.candidate.request.category]
        categories.extend(a.category for a in analysis.candidate.assertions)
        categories.extend(c.category for c in analysis.clues)
        if any(c is not None and c != self._category for c in categories):
            raise CommitError("category")
        if self._category != "phone" and (
            analysis.attribute_request is not None or analysis.attribute_changes
        ):
            raise CommitError("category_attributes")

    def _known_subjects(self) -> tuple[Subject, ...]:
        """仅发送已接受的人物来源，不将草案人物登记或按称呼自动合并。"""
        subjects = [Subject("other", p.references) for p in self._people]
        subjects.extend(c.subject for c in self._clues if c.subject.kind == "other")
        return tuple(dict.fromkeys(subjects))

    def _analyze_with_role(
        self,
        turns: tuple[Turn, ...],
        asked: Turn | None,
        complete: Complete,
        as_of: datetime,
        policy_revision: str,
        *,
        expose_draft: bool = True,
    ) -> InteractionCandidate:
        """草案不产生提案；同一快照中的最终结果才有机会进入宿主。"""
        try:
            draft = inspect_draft(
                turns,
                complete=complete,
                pending_question=asked,
                known_subjects=self._known_subjects(),
                category=self._category,
            )
            self._analysis_calls += (draft.call,)
            self._fresh(policy_revision)
            if draft.analysis is None:
                # 坏草案不能决定哪些旧记忆相关；独立角色只读实际对话。
                view = MemoryView(
                    self._user_id,
                    self._session_id,
                    as_of,
                    self._snapshot,
                    (),
                    False,
                    self._category,
                )
            else:
                active = (
                    self._store.active_memories(
                        self._user_id,
                        category=self._category,
                        as_of=as_of,
                        max_memory_age=self._max_age,
                    )
                    if self._use_policy.allows_personalisation(self._purpose)
                    else ()
                )
                view = build_memory_view(
                    user_id=self._user_id,
                    session_id=self._session_id,
                    as_of=as_of,
                    snapshot_token=self._snapshot,
                    active_memories=active,
                    draft=draft.analysis,
                    aliases=self._store.entity_aliases(),
                    category=self._category,
                )
            # 读取视图期间的外部修改，也必须在第二次发送前发现。
            self._fresh(policy_revision)
            task = AnalystTask(
                self._user_id,
                self._session_id,
                as_of,
                turns,
                draft.raw,
                view,
                asked,
                self._known_subjects(),
                draft.error_code,
                expose_draft=expose_draft,
                aliases=self._store.entity_aliases(),
            )
            final = MemoryAnalyst(complete).analyze(task)
            self._analysis_calls += (final.call,)
            return final.analysis
        except AnalystError as error:
            if error.call is not None:
                self._analysis_calls += (error.call,)
            raise
        finally:
            # 供应商错误与删除同时发生，也不能把旧原话留给下一轮重放。
            self._fresh(policy_revision)

    def _validate_clues(self, clues: tuple[PreferenceClue, ...]) -> None:
        """线索绕过长期提案层，仍须逐条检查原始本人证据的删除水位。"""
        known = {turn.turn_id: turn for turn in self._turns}
        for clue in clues:
            # 本人实体的删除不撤销亲属的独立表达；未知主体按保守本人边界检查。
            if clue.subject.kind == "other":
                continue
            if any(
                self._store.evidence_is_deleted(
                    self._user_id,
                    category=clue.category,
                    entity=clue.entity,
                    occurred_at=known[span.turn_id].occurred_at,
                    available_at=known[span.turn_id].available_at,
                )
                for span in clue.evidence + clue.subject.references
            ):
                # 不能只丢候选后留下旧原话，供下一轮模型再次引用。
                self._turns = ()
                self._clues = ()
                self._people = ()
                self._invalidate()
                raise CommitError("deleted_evidence")

    def _confirm(
        self, analysis: InteractionCandidate, turn: Turn, asked: Turn | None, as_of: datetime
    ) -> CommitPreview:
        confirmation = analysis.confirmation
        assert confirmation is not None
        if asked is None or self._pending is None or confirmation.question_turn_id != asked.turn_id:
            raise CommitError("question")
        choice = confirmation.choice
        if choice.outcome == "unclear":
            return self._pending
        if choice.outcome == "cancel":
            self._host.cancel(self._pending.ticket_id)
            return self._host.prepare(
                analysis.candidate, turns=self._turns, as_of=as_of, people=self._people
            )
        text = turn.content.strip().rstrip("。.!！?？,，").lower()
        if text in _AMBIGUOUS:
            raise CommitError("ambiguous_reply")
        if text in {"前者", "后者"} and (
            asked.content not in _SCOPE_QUESTIONS
            or choice.subject != "self"
            or choice.scope != ("current" if text == "前者" else "long_term")
            or any(c.beneficiary_id != "self" for c in self._pending.proposal.session_preferences)
        ):
            raise CommitError("ambiguous_reply")
        return self._host.confirm(
            self._pending.ticket_id,
            reply=turn,
            question=asked,
            subject=choice.subject,
            scope="current" if choice.subject == "other" else choice.scope,
            person_reference=choice.person_reference,
            as_of=as_of,
        )

    def _background(
        self,
        analysis: InteractionCandidate,
        preview: CommitPreview,
        as_of: datetime,
        *,
        is_recommendation: bool = False,
    ) -> tuple[PreferenceMemory, ...]:
        if is_recommendation or "recommend" in analysis.intents:
            return preview.proposal.decision.memories
        if not {"consult", "compare"}.intersection(analysis.intents):
            return ()
        if not self._use_policy.allows_personalisation(self._purpose):
            return ()
        if analysis.candidate.request.beneficiary.kind == "other":
            return ()
        memories = self._store.active_memories(
            self._user_id, category=self._category, as_of=as_of, max_memory_age=self._max_age
        )
        # 原文话题与存储键分开：只接受完整实体，或完整实体后紧接类别词“手机”。
        # 已有精确键优先，仅接受可信表的整实体别名，不对子串或更长商品短语猜测身份。
        known = {m.entity for m in memories}
        topics = {self._store.canonical_entity(self._category, t) for t in analysis.topic_entities}
        matched = topics & known
        matched.update(
            self._store.canonical_entity(self._category, topic.removesuffix("手机"))
            for topic in topics - known
            if self._category == "phone"
            and topic.endswith("手机")
            and self._store.canonical_entity(self._category, topic.removesuffix("手机")) in known
        )
        return tuple(
            m
            for m in memories
            if m.relation != "neutral" and (not analysis.topic_entities or m.entity in matched)
        )

    @staticmethod
    def _action(
        analysis: InteractionCandidate,
        preview: CommitPreview,
        *,
        is_recommendation: bool = False,
    ) -> ResponseAction:
        if preview.action == "clarify":
            return "clarify"
        if "feedback" in analysis.intents:
            return "feedback_target_required"
        if is_recommendation or "recommend" in analysis.intents:
            return "recommend"
        if "compare" in analysis.intents:
            return "compare"
        if "unrelated" in analysis.intents:
            return "continue"
        return "answer"

    def _get(self, ticket_id: str, as_of: datetime | None = None) -> _InteractionTicket:
        self._fresh()
        current = self._current
        if current is None or current.preview.ticket_id != ticket_id:
            raise CommitError("ticket")
        if as_of is not None and _utc(as_of, "as_of") < current.created_at:
            raise CommitError("time")
        return current

    def commit(self, ticket_id: str, *, as_of: datetime) -> CommitOutcome:
        """整轮接受成功后才发布线索；存储失败不会先发布半轮工作状态。"""
        ticket = self._get(ticket_id, as_of)
        self._validate_clues(ticket.analysis.clues)
        outcome = self._host.commit(ticket.preview.memory_preview.ticket_id, as_of=as_of)
        self._snapshot = self._store.snapshot_token(self._user_id)
        if not ticket.accepted:
            strong = {
                (a.subject.kind, a.category, self._store.canonical_entity(a.category, a.entity))
                for a in ticket.analysis.candidate.assertions
                if a.relation != "none"
            }
            # 确认候选自身没有新断言，须从已核验的提案目标识别提升后的本人事实。
            strong.update(
                ("self", m.category, m.entity)
                for m in ticket.preview.memory_preview.proposal.proposed_memories
            )
            other_strong = {
                (
                    a.subject.references,
                    a.category,
                    self._store.canonical_entity(a.category, a.entity),
                )
                for a in ticket.analysis.candidate.assertions
                if a.subject.kind == "other" and a.relation != "none"
            }
            clues = [
                c
                for c in self._clues
                if not (
                    c.subject.kind == "self"
                    and ("self", c.category, self._store.canonical_entity(c.category, c.entity))
                    in strong
                    or c.subject.kind == "other"
                    and (
                        c.subject.references,
                        c.category,
                        self._store.canonical_entity(c.category, c.entity),
                    )
                    in other_strong
                )
            ]
            for clue in ticket.analysis.clues:
                clues = [
                    c
                    for c in clues
                    if (
                        c.subject.kind,
                        c.subject.references if c.subject.kind == "other" else (),
                        c.category,
                        self._store.canonical_entity(c.category, c.entity),
                    )
                    != (
                        clue.subject.kind,
                        clue.subject.references if clue.subject.kind == "other" else (),
                        clue.category,
                        self._store.canonical_entity(clue.category, clue.entity),
                    )
                ]
                clues.append(clue)
            self._clues = tuple(clues)
            self._people = ticket.preview.memory_preview.proposal.people
            self._current = _InteractionTicket(
                ticket.preview, ticket.analysis, ticket.created_at, accepted=True
            )
        return outcome

    def cancel(self, ticket_id: str) -> None:
        """取消未接受的提案，保留实际发生的对话和已经接受的工作线索。"""
        self._get(ticket_id)
        self._drop_preview()

    def session_attribute_preferences(self) -> tuple[ContextAttributePreference, ...]:
        """通过同一fresh门控取得已接受他人会话属性，不能跨会话搬用。"""
        self._fresh()
        return self._host.session_attributes()
