"""本人长期属性的类型化证据；与品牌记录共享发布点而不混用语义。"""

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from math import isfinite
from typing import Self

from memrecagent.memory import _text, _utc
from memrecagent.product_attributes import validate_predicate


@dataclass(frozen=True)
class AttributeMemory:
    """active=False撤销同方向限制；硬软转换需分别撤销和新建，不能偷换修订链。"""

    memory_id: str
    user_id: str = field(repr=False)
    category: str
    attribute: str
    operator: str
    value: float | str | bool = field(repr=False)
    unit: str
    is_hard: bool
    active: bool
    evidence_ref: str = field(repr=False)
    evidence_text: str = field(repr=False)
    occurred_at: datetime
    available_at: datetime
    confidence: float | None = None
    subject_id: str = field(default="self", repr=False)
    supersedes: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "memory_id",
            "user_id",
            "category",
            "evidence_ref",
            "evidence_text",
            "subject_id",
        ):
            _text(getattr(self, name), name)
        if len(self.user_id) > 128 or self.subject_id != "self":
            raise ValueError("长期属性只接受有效用户的本人记录")
        if type(self.is_hard) is not bool or type(self.active) is not bool:
            raise ValueError("硬软标志与生效标志需要布尔值")
        validate_predicate(self.attribute, self.operator, self.value, self.unit)
        if self.confidence is not None and (
            type(self.confidence) not in (int, float)
            or not isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("confidence: 需要有限0至1数值或None")
        for name in ("occurred_at", "available_at"):
            object.__setattr__(self, name, _utc(getattr(self, name), name))
        if self.available_at < self.occurred_at:
            raise ValueError("available_at: 不能早于发生时间")
        if self.supersedes is not None:
            _text(self.supersedes, "supersedes")
            if self.supersedes == self.memory_id:
                raise ValueError("supersedes: 不能修订自身")

    def to_dict(self) -> dict:
        """显式类型判别，旧品牌schema保持逐字段兼容。"""
        return asdict(self) | {
            "record_type": "attribute",
            "schema_version": 1,
            "occurred_at": self.occurred_at.isoformat(),
            "available_at": self.available_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, value: dict) -> Self:
        """未知字段或类型均拒绝，不把损坏属性降级成可忽略内容。"""
        known = {f.name for f in fields(cls)} | {"record_type", "schema_version"}
        if not isinstance(value, dict) or value.keys() != known:
            raise ValueError("属性记忆字段缺失或未知")
        if (
            type(value["schema_version"]) is not int
            or value["schema_version"] != 1
            or value["record_type"] != "attribute"
        ):
            raise ValueError("不支持的属性记忆版本或类型")
        data = {k: v for k, v in value.items() if k not in {"record_type", "schema_version"}}
        for name in ("occurred_at", "available_at"):
            if not isinstance(data[name], str):
                raise ValueError("属性记忆时间格式无效")
            data[name] = datetime.fromisoformat(data[name])
        return cls(**data)
