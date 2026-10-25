"""实际记忆分析/受控提交到商品排序；不把模型草案当可信请求。"""

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta
from hashlib import sha256
from typing import Literal
from uuid import uuid4

from memrecagent.attribute_candidates import ContextAttributePreference
from memrecagent.deepseek_client import ModelReply
from memrecagent.memory import _text, _utc
from memrecagent.memory_commit import CommitError
from memrecagent.memory_extraction import Turn
from memrecagent.memory_feedback import FeedbackTarget, HostFeedbackSession
from memrecagent.memory_feedback_store import JsonFeedbackStore
from memrecagent.memory_interaction import ConversationMemorySession
from memrecagent.memory_products import MemoryProductRecommender, ProductEntityFact
from memrecagent.memory_store import JsonlMemoryStore
from memrecagent.memory_use_policy import JsonMemoryUseStore
from memrecagent.product_attributes import RequestConstraint, RequestRelaxation
from memrecagent.product_service import ProductService
from memrecagent.request_attributes import (
    CurrentAttributeRequest,
    advance_request,
    bind_constraints,
    bind_relaxations,
)
from memrecagent.request_soft_attributes import advance_soft_request
from memrecagent.result_count import resolve_result_count


class ConversationProductSession:
    """用户/会话由宿主绑定；会话人物只活在本实例，长期状态由旧守卫提交。"""

    def __init__(
        self,
        service: ProductService,
        store: JsonlMemoryStore,
        facts: tuple[ProductEntityFact, ...],
        *,
        facts_bundle_id: str,
        user_id: str,
        session_id: str,
        category: str = "phone",
    ) -> None:
        self._category = _text(category, "category")
        if len(category) > 128 or not any(f.category == category for f in facts):
            raise ValueError("category: 目录无匹配品类")
        self._service = service
        self._ranker = MemoryProductRecommender(
            service, store, facts, facts_bundle_id=facts_bundle_id
        )
        self._memory = ConversationMemorySession(
            store, user_id=user_id, session_id=session_id, category=category
        )
        self._user_id, self._session_id = user_id, session_id
        self._memory_root = store.root.resolve()
        self.last_analysis_calls: list[dict] = []
        self.last_new_memory_ids: list[str] = []
        self._attribute_request: CurrentAttributeRequest | None = None
        self._request_soft_preferences: tuple[ContextAttributePreference, ...] = ()
        self._last_request: dict | None = None
        self._last_request_state: str | None = None
        self._last_request_token: str | None = None
        self._last_count_metadata: dict | None = None
        self._feedback = HostFeedbackSession(
            JsonFeedbackStore(store.root / "feedback"), user_id=user_id, session_id=session_id
        )
        self._runtime_revision = 0
        self._feedback_overlay: dict[str, FeedbackTarget] = {}
        self._feedback_actions: dict[str, dict] = {}
        self._feedback_display: dict | None = None
        self._last_ranking: dict | None = None
        self._last_observed_at: datetime | None = None

    def process_user(
        self,
        turn: Turn,
        *,
        complete: Callable[[list[dict[str, str]]], ModelReply],
        as_of: datetime,
        strategy: str,
        k: int = 10,
        analysis_mode: Literal["direct", "analyst", "analyst_blind"] = "analyst",
    ) -> dict:
        """明确偏好可独立保存；不明人物/范围必须先追问，不能部分写入。"""
        self._clear_feedback_display(clear_request=True)
        self.last_analysis_calls = []
        self.last_new_memory_ids = []
        self._last_request = None
        self._last_request_state = None
        self._last_request_token = None
        self._last_count_metadata = None
        if sum(len(t.content) for t in self._memory.turns) + len(turn.content) > 24000:
            raise CommitError("session_limit")
        if (
            strategy not in self._service.health()["strategies"]
            or type(k) is not int
            or not 1 <= k <= 50
        ):
            raise ValueError("推荐配置无效")
        try:
            preview = self._memory.prepare_user(
                turn, complete=complete, as_of=as_of, analysis_mode=analysis_mode
            )
        except CommitError as error:
            if error.code in {"stale", "binding", "purpose", "deleted_evidence"}:
                self._clear_attribute_request()
            raise
        finally:
            self.last_analysis_calls = [asdict(call) for call in self._memory.last_analysis_calls]
        decision = preview.memory_preview.proposal.decision
        # 第一轮主模型仅强制进工具；历史只在fresh核验后送给第二轮，避免重放已删除证据。
        previous = self._memory.turns[:-1]
        context = []
        context_bytes = 2
        for old in reversed(previous):
            row = {"turn_id": old.turn_id, "role": old.role, "content": old.content}
            size = len(json.dumps(row, ensure_ascii=False).encode()) + 1
            if len(context) >= 12 or context_bytes + size > 12000:
                break
            context.append(row)
            context_bytes += size
        context.reverse()
        result = {
            "response_action": preview.response_action,
            "clarification": decision.clarification,
            "new_memory_ids": [],
            "ranking": None,
            "analysis_calls": [asdict(call) for call in preview.analysis_calls],
            "context_memories": [
                {
                    "entity": m.entity,
                    "category": m.category,
                    "relation": m.relation,
                    "evidence_ref": m.evidence_ref,
                }
                for m in preview.context_memories
            ],
            "state_token": self.state_token(),
            "conversation_context": context,
            "history_truncated": len(context) < len(previous),
        }
        if preview.response_action == "clarify":
            return result
        count = (
            resolve_result_count(turn.content) if preview.response_action == "recommend" else None
        )
        if count is not None and count.clarification is not None:
            notice = (
                "这条消息里的新长期偏好尚未保存。"
                if preview.memory_preview.proposal.proposed_memories
                else ""
            )
            result.update(response_action="clarify", clarification=notice + count.clarification)
            return result
        effective_k = count.count if count is not None and count.count is not None else k
        updated_attributes = advance_request(
            self._attribute_request,
            preview.attribute_request,
            user_id=self._user_id,
            session_id=self._session_id,
            turns=self._memory.turns,
            beneficiary_id=decision.beneficiary_id,
            is_recommendation=preview.response_action == "recommend",
        )
        updated_attributes, updated_soft = advance_soft_request(
            self._attribute_request,
            updated_attributes,
            self._request_soft_preferences,
            preview.memory_preview.proposal.session_attributes,
            candidate=preview.attribute_request,
            turns=self._memory.turns,
            beneficiary_id=decision.beneficiary_id,
            is_recommendation=preview.response_action == "recommend",
        )
        constraints = bind_constraints(
            updated_attributes, user_id=self._user_id, session_id=self._session_id
        )
        relaxations = bind_relaxations(
            updated_attributes, user_id=self._user_id, session_id=self._session_id
        )
        outcome = self._memory.commit(preview.ticket_id, as_of=as_of)
        if (self._attribute_request, self._request_soft_preferences) != (
            updated_attributes,
            updated_soft,
        ):
            self._runtime_revision += 1
        self._attribute_request = updated_attributes
        self._request_soft_preferences = updated_soft
        result["new_memory_ids"] = list(outcome.memory_ids)
        self.last_new_memory_ids = list(outcome.memory_ids)
        result["state_token"] = self.state_token()
        if preview.response_action == "recommend":
            request_id = updated_attributes.request_id if updated_attributes else uuid4().hex
            current_keys = {(c.attribute, c.operator) for c in constraints}
            removed = {r.attribute for r in relaxations}
            request_soft_keys = {(c.beneficiary_id, c.category, c.attribute) for c in updated_soft}
            attributes = {
                (c.beneficiary_id, c.category, c.attribute, c.operator, c.is_hard): c
                for c in (
                    *self._memory.session_attribute_preferences(),
                    *(
                        c
                        for c in preview.memory_preview.proposal.session_attributes
                        if c.retention != "request"
                    ),
                    *updated_soft,
                )
                if c.is_hard
                or c.retention == "request"
                or (c.beneficiary_id, c.category, c.attribute) not in request_soft_keys
            }
            bound_attributes = tuple(
                (
                    c.is_hard,
                    RequestConstraint(
                        self._user_id,
                        self._session_id,
                        c.beneficiary_id,
                        c.attribute,
                        c.operator,
                        c.value,
                        c.unit,
                        c.evidence_ref,
                        c.available_at,
                        request_id,
                    ),
                )
                for c in attributes.values()
                if c.beneficiary_id == decision.beneficiary_id
                and c.active
                and c.category == self._category
                and (c.is_hard or c.attribute not in {p.attribute for p in constraints})
                and c.available_at < as_of
                and (c.attribute, c.operator) not in current_keys
                and c.attribute not in removed
            )
            constraints += tuple(p for hard, p in bound_attributes if hard)
            soft_preferences = tuple(p for hard, p in bound_attributes if not hard)
            soft_relaxations = tuple(
                RequestRelaxation(
                    self._user_id,
                    self._session_id,
                    c.beneficiary_id,
                    request_id,
                    c.attribute,
                    c.evidence_ref,
                    c.available_at,
                )
                for c in updated_soft
                if not c.active
            )
            # 使用严格宿主已接受的人物和当前约束；不解析/信任草案或自由文本。
            query = dict(
                user_id=self._user_id,
                session_id=self._session_id,
                beneficiary_id=decision.beneficiary_id,
                category=self._category,
                as_of=as_of,
                strategy=strategy,
                k=effective_k,
                requested_entity=decision.requested_entity,
                context_preferences=decision.context_preferences,
                constraints=constraints,
                soft_preferences=soft_preferences,
                soft_relaxations=soft_relaxations,
                request_id=request_id,
                relaxations=relaxations,
            )
            result["ranking"] = self._rank(**query)
            if count is not None and count.count is not None:
                self._last_count_metadata = dict(
                    requested_count=count.count,
                    # 相同数量重复出现时一处来源足够，避免重复文本放大工具回执。
                    count_evidence=[
                        dict(turn_id=turn.turn_id, start=a, end=b) for a, b in count.spans[:1]
                    ],
                )
                result["ranking"].update(deepcopy(self._last_count_metadata))
            result["state_token"] = result["ranking"]["state_token"]
            self._last_ranking = deepcopy(result["ranking"])
            self._last_request = deepcopy(query)
            self._last_request_state = result["state_token"]
            self._last_request_token = uuid4().hex
        return result

    def recommendation_token(self) -> str:
        """记忆状态相同也可能是新问题；独立nonce只绑定最后实际接受的推荐请求。"""
        if (
            self._last_request is None
            or self._last_request_token is None
            or self.state_token() != self._last_request_state
        ):
            raise CommitError("stale_recommendation")
        return self._last_request_token

    def rank_current(
        self,
        *,
        use_long_term_memory: bool,
        strategy: str | None = None,
        request_token: str | None = None,
    ) -> dict:
        """可信宿主按最后实际接受的请求作只读对照，不能用gold重建预算/人物。"""
        bound_token = self.recommendation_token()
        if request_token is not None and request_token != bound_token:
            raise CommitError("stale_recommendation")
        query = deepcopy(self._last_request)
        query["use_long_term_memory"] = use_long_term_memory
        if strategy is not None:
            query["strategy"] = strategy
        result = self._rank(**query)
        if self._last_count_metadata is not None:
            result.update(deepcopy(self._last_count_metadata))
        if (
            result["state_token"] != self._last_request_state
            or self.state_token() != self._last_request_state
            or self._last_request_token != bound_token
        ):
            raise CommitError("stale_recommendation")
        return result

    def observe_assistant(self, turn: Turn) -> None:
        """登记实际已交付的回复，只有准确问题才能绑定后续确认。"""
        self._memory.observe_assistant(turn)
        self._last_observed_at = turn.available_at

    def state_token(self) -> str:
        """最终回复前由宿主检查删除/用途变化；不返回原始证据。"""
        disk = self._ranker.state_token(self._user_id)
        return (
            disk
            if not self._runtime_revision
            else sha256(f"{disk}:{self._runtime_revision}".encode()).hexdigest()
        )

    def _rank(self, **query) -> dict:
        before = self.state_token()
        result = self._ranker.recommend(**query)
        if self.state_token() != before:
            raise CommitError("stale_recommendation")
        result["state_token"] = before
        return result

    def _clear_feedback_display(self, *, clear_request: bool = False) -> None:
        self._feedback_display = None
        self._last_ranking = None
        self._last_observed_at = None
        if clear_request:
            if self._feedback_overlay:
                self._runtime_revision += 1
            self._feedback_overlay.clear()
            self._feedback.start_request()
            self._feedback_actions = {
                k: v for k, v in self._feedback_actions.items() if v["scope"] == "persistent"
            }

    def register_feedback_display(self, request_token: str, *, displayed_at: datetime) -> dict:
        """可信调用方仅在实际显示和observe之后登记；不接收任意商品ID。"""
        token = self.recommendation_token()
        displayed_at = _utc(displayed_at, "displayed_at")
        if (
            request_token != token
            or self._last_ranking is None
            or self._last_request is None
            or self._last_observed_at is None
            or self._last_observed_at < self._last_request["as_of"]
            or displayed_at < self._last_observed_at
        ):
            raise CommitError("feedback_display_required")
        if self._feedback_display is None:
            targets = []
            for item in self._last_ranking["items"]:
                targets.append(
                    {
                        kind: self._feedback.register_display(
                            FeedbackTarget(
                                kind,
                                item["category"],
                                item["item_id"] if kind == "product" else item["entity"],
                            ),
                            displayed_at=displayed_at,
                        )
                        for kind in ("product", "brand")
                    }
                )
            self._feedback_display = dict(
                token=uuid4().hex, request_token=token, targets=targets, displayed_at=displayed_at
            )
        return dict(
            display_token=self._feedback_display["token"],
            n_items=len(self._feedback_display["targets"]),
        )

    def _feedback_policy(self, scope: str) -> None:
        if scope not in ("persistent", "request") or self._last_request is None:
            raise CommitError("feedback_scope")
        beneficiary = self._last_request["beneficiary_id"]
        if beneficiary is None:
            raise CommitError("feedback_subject")
        policy = JsonMemoryUseStore(self._memory_root / "use-policy").get(self._user_id)
        if scope == "persistent" and (
            beneficiary != "self"
            or not policy.long_term_updates
            or not policy.allows_personalisation("recommendation")
        ):
            raise CommitError("feedback_purpose")

    def _feedback_rerank(self, *, event_id: str, action: str, scope: str, at: datetime) -> dict:
        query = deepcopy(self._last_request)
        query["as_of"] = at + timedelta(microseconds=1)
        query["request_feedback"] = tuple(dict.fromkeys(self._feedback_overlay.values()))
        self._clear_feedback_display()
        # 写入已发生时旧nonce无论重排是否成功都不可再被利用。
        self._last_request_token = None
        ranking = self._rank(**query)
        if self._last_count_metadata:
            ranking.update(deepcopy(self._last_count_metadata))
        self._last_request = query
        self._last_request_state = ranking["state_token"]
        self._last_request_token = uuid4().hex
        self._last_ranking = deepcopy(ranking)
        return dict(
            event_id=event_id,
            action=action,
            scope=scope,
            ranking=ranking,
            state_token=ranking["state_token"],
            request_token=self._last_request_token,
        )

    def submit_feedback(
        self, display_token: str, *, index: int, kind: str, scope: str, event_id: str, at: datetime
    ) -> dict:
        """位置由最近实际展示绑定；该入口不是LLM工具。"""
        token = self.recommendation_token()
        at = _utc(at, "at")
        _text(event_id, "event_id")
        display = self._feedback_display
        if (
            len(event_id) > 128
            or len(self._feedback_actions) >= 128
            or event_id in self._feedback_actions
            or display is None
            or display_token != display["token"]
            or token != display["request_token"]
            or type(index) is not int
            or not 1 <= index <= len(display["targets"])
            or kind not in ("product", "brand")
            or at < display["displayed_at"]
        ):
            raise CommitError("feedback_target")
        self._feedback_policy(scope)
        item = self._last_ranking["items"][index - 1]
        target = FeedbackTarget(
            kind, item["category"], item["item_id"] if kind == "product" else item["entity"]
        )
        self._feedback.submit(
            display["targets"][index - 1][kind], event_id=event_id, at=at, scope=scope
        )
        self._feedback_overlay[event_id] = target
        self._feedback_actions[event_id] = dict(scope=scope, active=True)
        self._runtime_revision += 1
        return self._feedback_rerank(event_id=event_id, action="blocked", scope=scope, at=at)

    def undo_feedback(self, event_id: str, *, action_id: str, at: datetime) -> dict:
        """只撤销本会话已知的用户动作；不从模型或任意目标构造写入。"""
        self.recommendation_token()
        at = _utc(at, "at")
        action = self._feedback_actions.get(event_id)
        if (
            action is None
            or not action["active"]
            or self._last_request is None
            or at <= self._last_request["as_of"]
        ):
            raise CommitError("feedback_event")
        self._feedback_policy(action["scope"])
        self._feedback.undo(event_id, action_id=action_id, at=at)
        action["active"] = False
        self._feedback_overlay.pop(event_id, None)
        self._runtime_revision += 1
        return self._feedback_rerank(
            event_id=event_id, action="undone", scope=action["scope"], at=at
        )

    def _clear_attribute_request(self) -> None:
        if self._attribute_request is not None or self._request_soft_preferences:
            self._runtime_revision += 1
        self._attribute_request = None
        self._request_soft_preferences = ()

    def refresh_read_state(self) -> str:
        """缓存读取也经过删除/用途/绑定守卫，失败时释放全部本会话原文引用。"""
        try:
            if self._ranker._store.root.resolve() != self._memory_root:
                raise CommitError("binding")
            self._memory._fresh()
            current = self.state_token()
            if self._last_request_state is not None and current != self._last_request_state:
                self._clear_feedback_display(clear_request=True)
                self._last_request = None
                self._last_request_state = None
                self._last_request_token = None
                self._last_count_metadata = None
            return self.state_token()
        except Exception:
            self._clear_feedback_display(clear_request=True)
            self._memory.clear_context()
            self._clear_attribute_request()
            self._last_request = None
            self._last_request_state = None
            self._last_request_token = None
            self._last_count_metadata = None
            self.last_new_memory_ids = []
            raise
