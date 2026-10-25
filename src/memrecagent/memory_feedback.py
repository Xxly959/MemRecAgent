"""可信展示目标的反馈接口；自然语言模型没有登记或写入权限。"""

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from memrecagent.memory import _text, _utc

if TYPE_CHECKING:
    from memrecagent.memory_feedback_store import JsonFeedbackStore


def _digest(value: str) -> str:
    _text(value, "identifier")
    return sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class FeedbackTarget:
    """精确目标，不隐式扩展商品、品牌或类别。"""

    kind: Literal["product", "brand"]
    category: str
    target_id: str

    def __post_init__(self) -> None:
        if self.kind not in ("product", "brand"):
            raise ValueError("反馈目标类型无效")
        _text(self.category, "category")
        _text(self.target_id, "target_id")

    def to_dict(self) -> dict:
        return {"kind": self.kind, "category": self.category, "target_id": self.target_id}

    @property
    def digest(self) -> str:
        from memrecagent.memory_feedback_store import _encode

        return _digest(_encode(self.to_dict()))


def _active(events: list[dict], target: FeedbackTarget, as_of: datetime) -> bool:
    """撤销只关闭所指当前事件，不恢复更旧屏蔽，也不撤销后来的点击。"""
    current = None
    active = False
    for event in events:
        if event["target"] != target.to_dict() or datetime.fromisoformat(event["at"]) >= as_of:
            continue
        if event["action"] == "block":
            current, active = event["event"], True
        elif event["ref"] == current:
            active = False
    return active


class HostFeedbackSession:
    """由可信宿主绑定用户和会话；注册表仅存本进程，单进程串行调用。"""

    def __init__(self, store: "JsonFeedbackStore", *, user_id: str, session_id: str) -> None:
        self.store = store
        _text(user_id, "user_id")
        self.user_id = user_id
        self.session_digest = _digest(session_id)
        self._displays: dict[str, tuple[FeedbackTarget, datetime]] = {}
        self._events: list[dict] = []
        self._receipts: dict[str, str] = {}

    def register_display(self, target: FeedbackTarget, *, displayed_at: datetime) -> str:
        """宿主展示后登记；该方法不能作为LLM工具暴露。"""
        if not isinstance(target, FeedbackTarget):
            raise ValueError("需要精确反馈目标")
        target = self.store.normalise_target(target)
        displayed_at = _utc(displayed_at, "displayed_at")
        token = uuid4().hex
        self._displays[token] = (target, displayed_at)
        return token

    def _check_source(self, target: FeedbackTarget, displayed_at: datetime) -> None:
        if self.store.is_deleted(self.user_id, target, displayed_at):
            raise ValueError("已删除的旧展示不能重放")

    def submit(
        self,
        display_id: str,
        *,
        event_id: str,
        at: datetime,
        scope: Literal["persistent", "request"] = "persistent",
    ) -> None:
        """对已展示目标提交屏蔽；重复同事件幂等，负载变化拒绝。"""
        if scope not in ("persistent", "request") or display_id not in self._displays:
            raise ValueError("反馈缺可信目标或范围无效")
        target, displayed_at = self._displays[display_id]
        at = _utc(at, "at")
        if displayed_at > at:
            raise ValueError("不能反馈未来展示")
        self._check_source(target, displayed_at)
        event = dict(
            event=_digest(event_id),
            action="block",
            target=target.to_dict(),
            display=_digest(display_id),
            session=self.session_digest,
            at=at.isoformat(),
            displayed_at=displayed_at.isoformat(),
            ref=None,
            scope=scope,
        )
        self._append(event)

    def _append(self, event: dict) -> None:
        from memrecagent.memory_feedback_store import _signature

        signature = _signature(event)
        persistent = self.store.receipt(self.user_id, event["event"])
        local = self._receipts.get(event["event"])
        if persistent is not None or local is not None:
            if (persistent or local) != signature:
                raise ValueError("反馈事件ID负载冲突")
            return
        if event["scope"] == "persistent":
            self.store.append(self.user_id, event)
        else:
            previous = [e for e in self._events if e["target"] == event["target"]]
            if previous and event["at"] <= previous[-1]["at"]:
                raise ValueError("反馈时间不能倒退或同刻覆盖")
            self._events.append(event)
            self._receipts[event["event"]] = signature

    def undo(self, event_id: str, *, action_id: str, at: datetime) -> None:
        """撤销已知屏蔽事件，不要求再次展示，不改变其原证据。"""
        at = _utc(at, "at")
        key = _digest(event_id)
        source = next((e for e in self._events if e["event"] == key), None)
        if source is None:
            source = self.store.find(self.user_id, key)
        if source is None or source["action"] != "block":
            raise ValueError("没有可撤销的屏蔽事件")
        target = FeedbackTarget(**source["target"])
        self._check_source(target, datetime.fromisoformat(source["displayed_at"]))
        if at <= datetime.fromisoformat(source["at"]):
            raise ValueError("撤销必须晚于屏蔽")
        event = dict(
            event=_digest(action_id),
            action="undo",
            target=source["target"],
            display=None,
            session=self.session_digest,
            at=at.isoformat(),
            displayed_at=None,
            ref=key,
            scope=source["scope"],
        )
        self._append(event)

    def blocked(
        self, target: FeedbackTarget, *, as_of: datetime, explicit_request: bool = False
    ) -> bool:
        """返回主动推荐屏蔽信号；明确主动请求仅本次覆盖，不擦除反馈。"""
        if not isinstance(target, FeedbackTarget) or type(explicit_request) is not bool:
            raise ValueError("反馈查询参数无效")
        as_of = _utc(as_of, "as_of")
        target = self.store.normalise_target(target)
        persistent = self.store.blocked(self.user_id, target, as_of)
        self._prune_deleted()
        return not explicit_request and (persistent or _active(self._events, target, as_of))

    def _prune_deleted(self) -> None:
        """外部管理删除后，下次访问丢弃本地派生原文；摘要收据不复活内容。"""
        self._events = [
            e
            for e in self._events
            if not self.store.is_deleted(
                self.user_id,
                FeedbackTarget(**e["target"]),
                datetime.fromisoformat(e["displayed_at"] or e["at"]),
            )
        ]
        self._displays = {
            key: value
            for key, value in self._displays.items()
            if not self.store.is_deleted(self.user_id, value[0], value[1])
        }

    def start_request(self) -> None:
        """宿主每次开始新推荐请求时调用，清空临时屏蔽和旧展示令牌。"""
        self._events.clear()
        self._displays.clear()
        # 收据继续留摘要，旧请求的event_id不能换负载重放。

    def delete(self, target: FeedbackTarget, *, at: datetime) -> int:
        """物理删除该目标历史，留下摘要水位防旧展示和事件复活。"""
        target = self.store.normalise_target(target)
        at = _utc(at, "at")
        local = [e for e in self._events if e["target"] == target.to_dict()]
        if any(datetime.fromisoformat(e["at"]) > at for e in local):
            raise ValueError("删除时间不能倒退")
        count = self.store.delete(self.user_id, target, at)
        self._prune_deleted()
        return count + len(local)
