"""类型化目录属性与宿主绑定的本次硬约束；未知值不能充当满足。"""

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Protocol

from memrecagent.memory import _text, _utc

NUMERIC_UNITS = {
    "price_cny": "CNY",
    "weight_g": "g",
    "battery_life_hours": "h",
    "screen_inches": "inch",
    "storage_gb": "GB",
}
UNITS = NUMERIC_UNITS | {"screen_type": "type", "eye_care": "bool"}


class AttributePredicate(Protocol):
    """长期和本次类型共用比较，只读取校验过的谓词字段。"""

    @property
    def attribute(self) -> str: ...
    @property
    def operator(self) -> str: ...
    @property
    def value(self) -> float | str | bool: ...


def _value(attribute: str, value: object, unit: str) -> None:
    if not isinstance(attribute, str) or attribute not in UNITS or unit != UNITS[attribute]:
        raise ValueError("属性或单位不支持，禁止自动换算")
    if attribute in NUMERIC_UNITS:
        valid = type(value) in (int, float) and isfinite(value) and value > 0
    elif attribute == "eye_care":
        valid = type(value) is bool
    else:
        valid = isinstance(value, str) and bool(value.strip()) and len(value) <= 64
    if not valid:
        raise ValueError("属性值类型无效")


def validate_predicate(attribute: str, operator: str, value: object, unit: str) -> None:
    """语言候选和可信请求使用同一属性类型及比较规则。"""
    _value(attribute, value, unit)
    if operator not in ("<=", ">=", "==") or (attribute not in NUMERIC_UNITS and operator != "=="):
        raise ValueError("属性比较方式无效")


@dataclass(frozen=True)
class ProductAttributeValue:
    """每项目录事实独立保留来源与可用时间，不能将未来规格提前使用。"""

    attribute: str
    value: float | str | bool
    unit: str
    evidence_ref: str
    available_at: datetime

    def __post_init__(self) -> None:
        _value(self.attribute, self.value, self.unit)
        _text(self.evidence_ref, "attribute_source")
        object.__setattr__(self, "available_at", _utc(self.available_at, "available_at"))


@dataclass(frozen=True)
class RequestConstraint:
    """绑定本次谓词；constraints硬过滤、soft_preferences软排序，不自动写长期。"""

    user_id: str
    session_id: str
    beneficiary_id: str
    attribute: str
    operator: str
    value: float | str | bool
    unit: str
    evidence_ref: str
    available_at: datetime
    request_id: str

    def __post_init__(self) -> None:
        for key in ("user_id", "session_id", "beneficiary_id", "request_id", "evidence_ref"):
            _text(getattr(self, key), key)
        validate_predicate(self.attribute, self.operator, self.value, self.unit)
        object.__setattr__(self, "available_at", _utc(self.available_at, "available_at"))


def validate_constraints(
    constraints: tuple[RequestConstraint, ...],
    *,
    user_id: str,
    session_id: str,
    beneficiary_id: str | None,
    as_of: datetime,
    request_id: str | None,
) -> None:
    """重复同方向限制须由宿主先解决，不将新旧预算默默取交集。"""
    if (
        not isinstance(constraints, tuple)
        or len(constraints) > 16
        or any(not isinstance(c, RequestConstraint) for c in constraints)
    ):
        raise ValueError("硬约束需要有限类型化元组")
    if len({(c.attribute, c.operator) for c in constraints}) != len(constraints):
        raise ValueError("同属性方向的硬约束重复，需先解析本次覆盖")
    if constraints or request_id is not None:
        _text(request_id, "request_id")
    if any(
        (c.user_id, c.session_id, c.beneficiary_id, c.request_id)
        != (user_id, session_id, beneficiary_id, request_id)
        or c.available_at >= as_of
        for c in constraints
    ):
        raise ValueError("硬约束人物/请求绑定或时间无效")


def decode_request_constraints(value: object) -> tuple[RequestConstraint, ...]:
    """工具与离线重放共用JSON边界，避免一条入口忽略或宽松解释约束。"""
    if not isinstance(value, list) or len(value) > 16:
        raise ValueError("本次约束需要有限列表")
    return tuple(
        RequestConstraint(**(c | {"available_at": datetime.fromisoformat(c["available_at"])}))
        for c in value
    )


@dataclass(frozen=True)
class RequestRelaxation:
    """宿主确认的本次解除，仅移除该属性的长期硬限制，不改变磁盘。"""

    user_id: str
    session_id: str
    beneficiary_id: str
    request_id: str
    attribute: str
    evidence_ref: str
    available_at: datetime

    def __post_init__(self) -> None:
        for key in ("user_id", "session_id", "beneficiary_id", "request_id", "evidence_ref"):
            _text(getattr(self, key), key)
        if not isinstance(self.attribute, str) or self.attribute not in UNITS:
            raise ValueError("本次解除的属性不支持")
        object.__setattr__(self, "available_at", _utc(self.available_at, "available_at"))


def validate_relaxations(
    relaxations: tuple[RequestRelaxation, ...],
    *,
    constraints: tuple[RequestConstraint, ...],
    user_id: str,
    session_id: str,
    beneficiary_id: str | None,
    request_id: str | None,
    as_of: datetime,
) -> None:
    """解除不能跨请求，也不能与本次新条件对同一属性自相矛盾。"""
    if (
        not isinstance(relaxations, tuple)
        or len(relaxations) > 16
        or any(not isinstance(r, RequestRelaxation) for r in relaxations)
    ):
        raise ValueError("本次解除需要有限类型化元组")
    if len({r.attribute for r in relaxations}) != len(relaxations):
        raise ValueError("本次解除属性重复")
    if any(
        (r.user_id, r.session_id, r.beneficiary_id, r.request_id)
        != (user_id, session_id, beneficiary_id, request_id)
        or r.available_at >= as_of
        for r in relaxations
    ):
        raise ValueError("本次解除人物/请求绑定或时间无效")
    if {r.attribute for r in relaxations} & {c.attribute for c in constraints}:
        raise ValueError("同一属性不能同时解除和施加本次条件")


def decode_request_relaxations(value: object) -> tuple[RequestRelaxation, ...]:
    """正式工具与回放共享严格解码，字段由构造器完整验证。"""
    if not isinstance(value, list) or len(value) > 16:
        raise ValueError("本次解除需要有限列表")
    return tuple(
        RequestRelaxation(**(r | {"available_at": datetime.fromisoformat(r["available_at"])}))
        for r in value
    )


def constraint_failure(
    attributes: tuple[ProductAttributeValue, ...],
    constraints: tuple[AttributePredicate, ...],
    *,
    as_of: datetime,
) -> tuple[str, str] | None:
    """只报告第一个失败原因；缺失/未来为unknown，不推断默认值。"""
    known = {a.attribute: a for a in attributes if a.available_at < as_of}
    for c in constraints:
        fact = known.get(c.attribute)
        if fact is None:
            return "attribute_unknown", c.attribute
        if c.operator == "==":
            matched = fact.value == c.value
        elif c.operator == "<=":
            matched = fact.value <= c.value
        else:
            matched = fact.value >= c.value
        if not matched:
            return "attribute_constraint", c.attribute
    return None
