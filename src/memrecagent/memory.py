"""偏好证据快照：归属用户、偏好对象和修订关系分别保存。"""

from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import Literal, Self


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}: 需要非空字符串")
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise ValueError(f"{name}: 文本编码无效") from None
    return value


def _utc(value: object, name: str) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError(f"{name}: 需要带时区时间")
    try:
        return value.astimezone(UTC)
    except (OverflowError, ValueError):
        raise ValueError(f"{name}: UTC时间超出支持范围") from None


def _memory_age(value: timedelta | None) -> timedelta | None:
    """默认不失效；可选年龄是调用方配置，不是推测出的用户事实。"""
    if value is not None and (not isinstance(value, timedelta) or value <= timedelta(0)):
        raise ValueError("max_memory_age: 需要正时间间隔或None")
    return value


@dataclass(frozen=True)
class PreferenceMemory:
    """单条证据不原地修改；偏好改变以新证据显式修订旧记录。"""

    memory_id: str
    user_id: str = field(repr=False)
    category: str
    entity: str
    relation: Literal["prefer", "dislike", "exclude", "neutral"]
    evidence_ref: str = field(repr=False)
    evidence_text: str = field(repr=False)
    occurred_at: datetime
    available_at: datetime
    confidence: float | None = None
    subject_id: str = field(default="self", repr=False)
    supersedes: str | None = None

    def __post_init__(self) -> None:
        for name in [
            "memory_id",
            "user_id",
            "category",
            "entity",
            "relation",
            "evidence_ref",
            "evidence_text",
            "subject_id",
        ]:
            _text(getattr(self, name), name)
        if len(self.user_id) > 128:
            raise ValueError("user_id: 长度超过128字符")
        if self.subject_id != "self":
            raise ValueError("subject_id: 长期记忆只接受本人偏好")
        if self.relation not in {"prefer", "dislike", "exclude", "neutral"}:
            raise ValueError("relation: 不支持的关系")
        if self.confidence is not None and (
            type(self.confidence) not in (float, int)
            or not 0 <= self.confidence <= 1
            or not isfinite(self.confidence)
        ):
            raise ValueError("confidence: 需要有限0至1数值或None")
        for name in ["occurred_at", "available_at"]:
            object.__setattr__(self, name, _utc(getattr(self, name), name))
        if self.available_at < self.occurred_at:
            raise ValueError("available_at: 不能早于发生时间")
        if self.supersedes is not None:
            _text(self.supersedes, "supersedes")
            if self.supersedes == self.memory_id:
                raise ValueError("supersedes: 不能修订自身")

    def to_dict(self) -> dict:
        """完整保留字段；时间统一UTC，版本与数值含义可追溯。"""
        return asdict(self) | {
            "schema_version": 1,
            "occurred_at": self.occurred_at.isoformat(),
            "available_at": self.available_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, value: dict) -> Self:
        """读取时重新校验，不静默猜测字段或填充人物身份。"""
        known = {item.name for item in fields(cls)} | {"schema_version"}
        if not isinstance(value, dict) or value.keys() != known:
            raise ValueError("记忆字段缺失或未知")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("不支持的记忆版本")
        data = {key: item for key, item in value.items() if key != "schema_version"}
        for name in ["occurred_at", "available_at"]:
            try:
                if not isinstance(data[name], str):
                    raise ValueError
                data[name] = datetime.fromisoformat(data[name])
            except ValueError:
                raise ValueError(f"{name}: 无效时间格式") from None
        return cls(**data)
