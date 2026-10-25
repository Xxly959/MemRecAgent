"""已有推荐分与已解析偏好的透明融合；不训练、不推断品牌、不写记忆。"""

import json
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from math import isfinite

from memrecagent.memory import _text, _utc
from memrecagent.memory_feedback import FeedbackTarget
from memrecagent.memory_feedback_store import JsonFeedbackStore
from memrecagent.memory_policy import ContextPreference, resolve_recommendation
from memrecagent.memory_store import JsonlMemoryStore
from memrecagent.memory_use_policy import JsonMemoryUseStore
from memrecagent.product_attributes import (
    ProductAttributeValue,
    RequestConstraint,
    RequestRelaxation,
    constraint_failure,
    validate_constraints,
    validate_relaxations,
)
from memrecagent.product_models import score_model
from memrecagent.product_service import ProductService
from memrecagent.soft_attributes import soft_attribute_signal


@dataclass(frozen=True)
class ProductEntityFact:
    """可信目录的精确实体映射；来源和可用时间必须由宿主提供。"""

    item_id: str
    category: str
    entity: str
    evidence_ref: str
    available_at: datetime
    attributes: tuple[ProductAttributeValue, ...] = ()

    def __post_init__(self) -> None:
        for key in ("item_id", "category", "entity", "evidence_ref"):
            _text(getattr(self, key), key)
        object.__setattr__(self, "available_at", _utc(self.available_at, "available_at"))
        if (
            not isinstance(self.attributes, tuple)
            or len(self.attributes) > 16
            or any(not isinstance(a, ProductAttributeValue) for a in self.attributes)
            or len({a.attribute for a in self.attributes}) != len(self.attributes)
        ):
            raise ValueError("商品属性必须是唯一、有限的类型化事实")


def decode_product_entity_fact(row: object) -> ProductEntityFact:
    """两种实际桥接入口共用严格目录解码，兼容旧的无属性目录。"""
    if not isinstance(row, dict):
        raise ValueError("映射条目需要对象")
    attributes = row.get("attributes", [])
    if not isinstance(attributes, list) or len(attributes) > 16:
        raise ValueError("属性事实需要有限列表")
    values = tuple(
        ProductAttributeValue(**(a | {"available_at": datetime.fromisoformat(a["available_at"])}))
        for a in attributes
    )
    return ProductEntityFact(
        **(
            row
            | {"available_at": datetime.fromisoformat(row["available_at"]), "attributes": values}
        )
    )


class MemoryProductRecommender:
    """先解析人物/请求策略，再对完整未见候选过滤及固定加权。"""

    def __init__(
        self,
        service: ProductService,
        store: JsonlMemoryStore,
        facts: tuple[ProductEntityFact, ...],
        *,
        facts_bundle_id: str,
    ) -> None:
        rows = tuple(facts)
        if not isinstance(service, ProductService) or not isinstance(store, JsonlMemoryStore):
            raise ValueError("需要已有商品服务及记忆存储")
        if facts_bundle_id != service.bundle_id or not all(
            isinstance(f, ProductEntityFact) for f in rows
        ):
            raise ValueError("实体事实与目录版本不匹配")
        if (
            len({f.item_id for f in rows}) != len(rows)
            or {f.item_id for f in rows} != service.counts.keys()
        ):
            raise ValueError("实体映射必须完整且每商品唯一")
        self._source_entities = {f.item_id: f.entity for f in rows}
        aliases = store.entity_aliases()
        self._aliases = aliases
        if aliases is not None:
            rows = tuple(replace(f, entity=aliases.canonical(f.category, f.entity)) for f in rows)
        self._service, self._store, self._facts = service, store, rows
        self._bundle_id = facts_bundle_id

    def state_token(self, user_id: str) -> str:
        """删除和许可ABA都使旧推荐快照失效，不返回原始用户证据。"""
        policy, revision = JsonMemoryUseStore(self._store.root / "use-policy").snapshot(user_id)
        state = [
            self._store.snapshot_token(user_id),
            asdict(policy),
            revision,
            JsonFeedbackStore(self._store.root / "feedback").snapshot_token(user_id),
        ]
        return sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()

    def recommend(
        self,
        *,
        user_id: str,
        session_id: str,
        beneficiary_id: str | None,
        category: str,
        as_of: datetime,
        strategy: str,
        k: int = 10,
        requested_entity: str | None = None,
        context_preferences: tuple[ContextPreference, ...] = (),
        change_scope: str = "request",
        soft_weight: float = 0.75,
        max_memory_age: timedelta | None = None,
        normalisation: str = "minmax",
        use_long_term_memory: bool = True,
        constraints: tuple[RequestConstraint, ...] = (),
        request_id: str | None = None,
        relaxations: tuple[RequestRelaxation, ...] = (),
        request_feedback: tuple[FeedbackTarget, ...] = (),
        soft_preferences: tuple[RequestConstraint, ...] = (),
        soft_relaxations: tuple[RequestRelaxation, ...] = (),
        use_soft_attributes: bool = True,
    ) -> dict:
        """归一化基础分加固定偏好项；默认兼容旧min-max，rank仅显式选择。"""
        if (
            not isinstance(request_feedback, tuple)
            or len(request_feedback) > 100
            or any(not isinstance(t, FeedbackTarget) for t in request_feedback)
            or len(set(request_feedback)) != len(request_feedback)
        ):
            raise ValueError("当前反馈必须由可信宿主提供有限唯一精确目标")
        if normalisation not in ("minmax", "rank"):
            raise ValueError("归一化只支持minmax或rank")
        if type(use_soft_attributes) is not bool:
            raise ValueError("软属性开关需要布尔值")
        if type(use_long_term_memory) is not bool:
            raise ValueError("长期记忆开关需要明确布尔值")
        if (
            type(soft_weight) not in (int, float)
            or not isfinite(soft_weight)
            or not 0 <= soft_weight <= 1
        ):
            raise ValueError("软权重必须是0到1的有限数")
        if type(k) is not int or not 1 <= k <= 50:
            raise ValueError("k必须是1到50的整数")
        as_of = _utc(as_of, "as_of")
        validate_constraints(
            constraints,
            user_id=user_id,
            session_id=session_id,
            beneficiary_id=beneficiary_id,
            as_of=as_of,
            request_id=request_id,
        )
        validate_constraints(
            soft_preferences,
            user_id=user_id,
            session_id=session_id,
            beneficiary_id=beneficiary_id,
            as_of=as_of,
            request_id=request_id,
        )
        validate_relaxations(
            soft_relaxations,
            constraints=soft_preferences,
            user_id=user_id,
            session_id=session_id,
            beneficiary_id=beneficiary_id,
            request_id=request_id,
            as_of=as_of,
        )
        validate_relaxations(
            relaxations,
            constraints=constraints,
            user_id=user_id,
            session_id=session_id,
            beneficiary_id=beneficiary_id,
            request_id=request_id,
            as_of=as_of,
        )
        if (
            self._service.bundle_id != self._bundle_id
            or {f.item_id for f in self._facts} != self._service.counts.keys()
        ):
            raise ValueError("绑定目录已改变")
        cutoff = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=self._service.as_of_ms)
        if as_of < cutoff:
            raise ValueError("查询时刻不能早于冻结模型截止")
        before = self.state_token(user_id)
        if beneficiary_id == "self" and any(
            self._store.attribute_evidence_is_deleted(
                user_id, category=category, attribute=p.attribute, available_at=p.available_at
            )
            for p in (*constraints, *soft_preferences, *relaxations, *soft_relaxations)
        ):
            raise ValueError("deleted_attribute_evidence")
        decision = resolve_recommendation(
            self._store,
            user_id=user_id,
            session_id=session_id,
            beneficiary_id=beneficiary_id,
            category=category,
            as_of=as_of,
            requested_entity=requested_entity,
            context_preferences=context_preferences,
            change_scope=change_scope,
            max_memory_age=max_memory_age,
            use_long_term_memory=use_long_term_memory,
        )
        requested_entity = decision.requested_entity
        result = {
            "status": decision.status,
            "bundle_id": self._bundle_id,
            "as_of_ms": self._service.as_of_ms,
            "query_at": as_of.isoformat(),
            "items": [],
            "filtered": [],
            "n_filtered": 0,
            "filter_counts": {},
            "filtered_truncated": False,
            "n_candidates": 0,
            "overridden_memory_ids": list(decision.overridden_memory_ids),
            "state_token": before,
            "soft_weight": float(soft_weight),
            "normalisation": normalisation,
            "use_long_term_memory": use_long_term_memory,
            "soft_attributes_enabled": use_soft_attributes,
        }
        if decision.status == "clarify":
            result["clarification"] = decision.clarification
            if self.state_token(user_id) != before:
                raise ValueError("request_state_changed")
            return result
        policy = JsonMemoryUseStore(self._store.root / "use-policy").get(user_id)
        all_attributes = (
            self._store.active_attribute_memories(
                user_id, category=category, as_of=as_of, max_memory_age=max_memory_age
            )
            if use_long_term_memory
            and beneficiary_id == "self"
            and policy.allows_personalisation("recommendation")
            else ()
        )
        feedback = (
            JsonFeedbackStore(self._store.root / "feedback").active_targets(user_id, as_of=as_of)
            if use_long_term_memory
            and beneficiary_id == "self"
            and policy.allows_personalisation("recommendation")
            else ()
        )
        blocked_products = {
            t.target_id for t in feedback if t.category == category and t.kind == "product"
        }
        blocked_brands = {
            t.target_id for t in feedback if t.category == category and t.kind == "brand"
        }
        attributes = tuple(m for m in all_attributes if m.is_hard)
        current_soft_keys = {
            c.attribute for c in (*constraints, *soft_preferences, *soft_relaxations)
        }
        old_soft = tuple(m for m in all_attributes if not m.is_hard)
        overridden_soft = tuple(m for m in old_soft if m.attribute in current_soft_keys)
        effective_soft = (
            (tuple(m for m in old_soft if m not in overridden_soft) + soft_preferences)
            if use_soft_attributes
            else ()
        )
        if overridden_soft:
            result["overridden_soft_attribute_memory_ids"] = [m.memory_id for m in overridden_soft]
        if soft_relaxations:
            result["ignored_soft_attributes"] = [r.attribute for r in soft_relaxations]
        if effective_soft:
            result["soft_attribute_preferences"] = [
                dict(
                    attribute=p.attribute,
                    operator=p.operator,
                    value=p.value,
                    unit=p.unit,
                    scope="long_term" if hasattr(p, "memory_id") else "current",
                    reference=p.memory_id if hasattr(p, "memory_id") else p.evidence_ref,
                )
                for p in effective_soft
            ]
        current_keys = {(c.attribute, c.operator) for c in constraints}
        relaxed = {r.attribute for r in relaxations}
        overridden = tuple(
            m
            for m in attributes
            if (m.attribute, m.operator) in current_keys or m.attribute in relaxed
        )
        effective_constraints = constraints + tuple(m for m in attributes if m not in overridden)
        if attributes:
            result["attribute_memory_ids"] = [m.memory_id for m in attributes]
            result["overridden_attribute_memory_ids"] = [m.memory_id for m in overridden]
        if effective_constraints:
            result["n_constraints"] = len(effective_constraints)
            result["constraint_policy"] = "fail_closed_unknown"

        # 借用旧服务的验证、冷用户路由；不能只重排其已截断Top-K。
        if strategy not in self._service.health()["strategies"]:
            raise ValueError("不支持的推荐策略，禁止静默切换")
        # 未建立独立受益人画像时，owner的CF向量和seen都不能代替对方。
        is_self = beneficiary_id == "self"
        route = self._service.recommend(
            user_id, bundle_id=self._bundle_id, strategy=strategy if is_self else "popularity", k=1
        )
        used = route["used_strategy"]
        raw = (
            self._service.counts
            if used == "popularity"
            else score_model(self._service.models[used], user_id.strip())
        )
        if raw.keys() != self._service.counts.keys():
            raise ValueError("基础分与目录不匹配")
        scores = {item: float(value) for item, value in raw.items()}
        if not all(isfinite(value) for value in scores.values()):
            raise ValueError("基础分必须有限")
        seen = self._service.seen.get(user_id.strip(), set()) if is_self else set()
        pool = [
            f
            for f in self._facts
            if f.item_id not in seen and f.category == category and f.available_at < as_of
        ]
        result["n_candidates"] = len(pool)
        result["used_strategy"] = used
        result["requested_strategy"] = strategy
        result["routing_reason"] = "owner_profile" if is_self else "non_self_public_baseline"
        result["base_score_kind"] = (
            "positive_reviewer_count" if used == "popularity" else "dot_product"
        )
        # 归一化范围在记忆过滤前固定，避免各组偷偷更改基础特征。
        low = min((scores[f.item_id] for f in pool), default=0.0)
        high = max((scores[f.item_id] for f in pool), default=0.0)
        scale = max(abs(low), abs(high))
        ranked_scores = sorted(scores[f.item_id] for f in pool) if normalisation == "rank" else []
        sources: dict[str, list[dict]] = {}
        for memory in decision.memories:
            sources.setdefault(memory.entity, []).append(
                {
                    "entity": memory.entity,
                    "relation": memory.relation,
                    "scope": "long_term",
                    "reference": memory.memory_id,
                    "evidence_ref": memory.evidence_ref,
                    "value": {"prefer": 1, "dislike": -1}.get(memory.relation, 0),
                }
            )
        for preference in decision.context_preferences:
            sources.setdefault(preference.entity, []).append(
                {
                    "entity": preference.entity,
                    "relation": preference.relation,
                    "scope": "current",
                    "reference": preference.evidence_ref,
                    "evidence_ref": preference.evidence_ref,
                    "value": {"prefer": 1, "dislike": -1}.get(preference.relation, 0),
                }
            )
        items = []
        for fact in pool:
            failure = constraint_failure(fact.attributes, effective_constraints, as_of=as_of)
            reason = (
                "requested_entity"
                if requested_entity is not None and fact.entity != requested_entity
                else ("excluded_entity" if fact.entity in decision.excluded_entities else None)
            )
            # 本次实际点击晚于原品牌请求，不能被同一旧请求反向覆盖；各消融组相同。
            if (
                reason is None
                and FeedbackTarget("product", category, fact.item_id) in request_feedback
            ):
                reason = "feedback_current_product"
            if (
                reason is None
                and FeedbackTarget("brand", category, fact.entity) in request_feedback
            ):
                reason = "feedback_current_brand"
            if reason is None and fact.item_id in blocked_products:
                reason = "feedback_product"
            if reason is None and fact.entity in blocked_brands and requested_entity != fact.entity:
                reason = "feedback_brand"
            if reason is None and failure is not None:
                reason = failure[0]
            if reason:
                result["n_filtered"] += 1
                result["filter_counts"][reason] = result["filter_counts"].get(reason, 0) + 1
                if len(result["filtered"]) < 20:
                    result["filtered"].append(
                        {"item_id": fact.item_id, "entity": fact.entity, "reason": reason}
                    )
                else:
                    result["filtered_truncated"] = True
                continue
            base = scores[fact.item_id]
            normalised = (
                0.0 if high == low else (base / scale - low / scale) / (high / scale - low / scale)
            )
            if normalisation == "rank" and high != low:
                # 同分取平均名次，归一化池在记忆过滤前固定；常量池保持全零。
                left = bisect_left(ranked_scores, base)
                right = bisect_right(ranked_scores, base)
                normalised = ((left + right - 1) / 2) / (len(ranked_scores) - 1)
            signals = [s.copy() for s in sources.get(fact.entity, ()) if s["value"]]
            signal = sum(s["value"] for s in signals)
            attribute_signal, attribute_signals = soft_attribute_signal(
                fact.attributes, effective_soft, as_of=as_of
            )
            metadata = self._service.products.get(fact.item_id, {})
            items.append(
                {
                    "item_id": fact.item_id,
                    "entity": fact.entity,
                    "category": fact.category,
                    "title": metadata.get("title"),
                    "entity_source": fact.evidence_ref,
                    **(
                        dict(
                            entity_alias_source=self._aliases.source_for(
                                fact.category, self._source_entities[fact.item_id]
                            )
                        )
                        if self._store.entity_registry_digest is not None
                        else {}
                    ),
                    "base_score": base,
                    "normalised_base_score": normalised,
                    "preference_signal": signal,
                    "score": normalised + float(soft_weight) * (signal + attribute_signal),
                    "signals": signals,
                    **(
                        dict(
                            attribute_preference_signal=attribute_signal,
                            attribute_signals=attribute_signals,
                        )
                        if effective_soft
                        else {}
                    ),
                }
            )
        items.sort(key=lambda item: (-item["score"], -item["base_score"], item["item_id"]))
        result["items"] = items[:k]
        result["empty_reason"] = "no_matching_candidates" if not items else None
        if self.state_token(user_id) != before:
            raise ValueError("request_state_changed")
        return result
