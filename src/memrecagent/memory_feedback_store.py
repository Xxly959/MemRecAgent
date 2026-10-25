"""版1精确目标反馈快照；单进程串行、严格校验及原子替换。"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from memrecagent.entity_aliases import EntityAliases, checked_entity_aliases, load_entity_aliases
from memrecagent.memory import _utc
from memrecagent.memory_feedback import FeedbackTarget, _active, _digest
from memrecagent.memory_management_guard import ensure_no_pending
from memrecagent.memory_store import _unique_keys


def _encode(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _signature(event: dict) -> str:
    return _digest(_encode(event))


def _hash(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _time(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("时间无效")
    result = _utc(datetime.fromisoformat(value), "at")
    if result.isoformat() != value:
        raise ValueError("需要规范UTC时间")
    return result


def _target(value: object) -> FeedbackTarget:
    if not isinstance(value, dict) or value.keys() != {"kind", "category", "target_id"}:
        raise ValueError("目标无效")
    return FeedbackTarget(**value)


class JsonFeedbackStore:
    """每用户一个文件；不存原始对话，身份和事件ID仅用摘要绑定。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        table = load_entity_aliases(self.root.parent)
        self.entity_registry_digest = table.digest if table is not None else None

    def entity_aliases(self) -> EntityAliases | None:
        return checked_entity_aliases(self.root.parent, self.entity_registry_digest)

    def normalise_target(self, target: FeedbackTarget) -> FeedbackTarget:
        if not isinstance(target, FeedbackTarget):
            raise ValueError("反馈目标无效")
        table = self.entity_aliases()
        if table is not None and target.kind == "brand":
            return FeedbackTarget(
                target.kind, target.category, table.canonical(target.category, target.target_id)
            )
        return target

    def _path(self, user_id: str) -> Path:
        return self.root / (_digest(user_id) + ".json")

    def _load(self, user_id: str) -> dict:
        ensure_no_pending(self.root.parent, user_id)
        table = self.entity_aliases()
        path = self._path(user_id)
        if path.is_symlink():
            raise ValueError("不接受反馈符号链接文件")
        try:
            raw = path.read_text()
        except FileNotFoundError:
            return dict(
                schema_version=2 if table is not None else 1,
                owner_digest=path.stem,
                events=[],
                receipts={},
                deleted={},
                **(dict(entity_registry_digest=table.digest) if table is not None else {}),
            )
        try:
            data = json.loads(raw, object_pairs_hook=_unique_keys)
        except (ValueError, UnicodeError):
            raise ValueError("反馈文件无效") from None
        return self._validate(user_id, data)

    def _validate(self, user_id: str, data: object) -> dict:
        """读取和写入共享完整契约，坏候选不能破坏正式文件。"""
        try:
            table = self.entity_aliases()
            if not isinstance(data, dict) or data.keys() != (
                {
                    "schema_version",
                    "owner_digest",
                    "events",
                    "receipts",
                    "deleted",
                }
                | ({"entity_registry_digest"} if table is not None else set())
            ):
                raise ValueError("反馈文件字段无效")
            if (
                type(data["schema_version"]) is not int
                or data["schema_version"] != (2 if table is not None else 1)
                or (table is not None and data["entity_registry_digest"] != table.digest)
                or data["owner_digest"] != _digest(user_id)
                or not isinstance(data["events"], list)
                or not isinstance(data["receipts"], dict)
                or not isinstance(data["deleted"], dict)
            ):
                raise ValueError("反馈版本或归属无效")
            if any(not _hash(k) or not _hash(v) for k, v in data["receipts"].items()):
                raise ValueError("幂等收据无效")
            deleted = {}
            for scope, value in data["deleted"].items():
                if not _hash(scope):
                    raise ValueError("删除水位无效")
                deleted[scope] = _time(value)
            known, latest = {}, {}
            for e in data["events"]:
                if not isinstance(e, dict) or e.keys() != {
                    "event",
                    "action",
                    "target",
                    "display",
                    "session",
                    "at",
                    "displayed_at",
                    "ref",
                    "scope",
                }:
                    raise ValueError("反馈事件字段无效")
                target = _target(e["target"])
                if (
                    table is not None
                    and target.kind == "brand"
                    and table.canonical(target.category, target.target_id) != target.target_id
                ):
                    raise ValueError("canonical_feedback_target_required")
                time = _time(e["at"])
                key = target.digest
                if (
                    not _hash(e["event"])
                    or not _hash(e["session"])
                    or e["event"] in known
                    or e["scope"] != "persistent"
                    or (key in latest and time <= latest[key])
                    or (key in deleted and time <= deleted[key])
                    or data["receipts"].get(e["event"]) != _signature(e)
                ):
                    raise ValueError("反馈时间或收据无效")
                if e["action"] == "block":
                    display_time = _time(e["displayed_at"])
                    if (
                        not _hash(e["display"])
                        or e["ref"] is not None
                        or display_time > time
                        or (key in deleted and display_time <= deleted[key])
                    ):
                        raise ValueError("反馈展示无效")
                elif e["action"] == "undo":
                    source = known.get(e["ref"]) if isinstance(e["ref"], str) else None
                    if (
                        source is None
                        or source["action"] != "block"
                        or source["target"] != e["target"]
                        or e["display"] is not None
                        or e["displayed_at"] is not None
                    ):
                        raise ValueError("撤销来源无效")
                else:
                    raise ValueError("反馈动作无效")
                known[e["event"]], latest[key] = e, time
            return data
        except (ValueError, TypeError, KeyError, UnicodeError):
            raise ValueError("反馈文件无效") from None

    def _write(self, user_id: str, data: dict) -> None:
        self.entity_aliases()
        path = self._path(user_id)
        self.root.mkdir(parents=True, exist_ok=True)
        temp = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.root, prefix=".feedback-", delete=False
            ) as f:
                temp = Path(f.name)
                f.write(_encode(data) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp, path)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)

    def snapshot_token(self, user_id: str) -> str:
        """完整合法反馈状态摘要；撤销/删除仍改变旧推荐的有效性。"""
        return _digest(_encode(self._load(user_id)))

    def active_targets(self, user_id: str, *, as_of: datetime) -> tuple[FeedbackTarget, ...]:
        """一次读取当前有效精确屏蔽，不暴露展示令牌或历史事件。"""
        as_of = _utc(as_of, "as_of")
        events = self._load(user_id)["events"]
        targets = {FeedbackTarget(**event["target"]) for event in events}
        return tuple(
            target
            for target in sorted(targets, key=lambda t: (t.kind, t.category, t.target_id))
            if _active(events, target, as_of)
        )

    def receipt(self, user_id: str, event_digest: str) -> str | None:
        """返回幂等负载摘要，不暴露已删除目标。"""
        return self._load(user_id)["receipts"].get(event_digest)

    def find(self, user_id: str, event_digest: str) -> dict | None:
        """撤销需真实尚存的源事件；删除后只剩收据，不能复活。"""
        return next((e for e in self._load(user_id)["events"] if e["event"] == event_digest), None)

    def append(self, user_id: str, event: dict) -> None:
        """调用方为可信会话；完整验证后才发布，失败没有半条事件。"""
        data = self._load(user_id)
        if not isinstance(event, dict) or not _hash(event.get("event")):
            raise ValueError("反馈事件无效")
        key, signature = event["event"], _signature(event)
        if key in data["receipts"]:
            if data["receipts"][key] != signature:
                raise ValueError("反馈事件ID负载冲突")
            return
        data["events"].append(event)
        data["receipts"][key] = signature
        self._validate(user_id, data)
        self._write(user_id, data)

    def is_deleted(self, user_id: str, target: FeedbackTarget, source_at: datetime) -> bool:
        target = self.normalise_target(target)
        cutoff = self._load(user_id)["deleted"].get(target.digest)
        return cutoff is not None and source_at <= _time(cutoff)

    def blocked(self, user_id: str, target: FeedbackTarget, as_of: datetime) -> bool:
        return _active(self._load(user_id)["events"], self.normalise_target(target), as_of)

    def delete(self, user_id: str, target: FeedbackTarget, at: datetime) -> int:
        """删除含目标原文的事件，保留不可逆摘要和删除水位。"""
        data = self._load(user_id)
        target = self.normalise_target(target)
        at = _utc(at, "at")
        matching = [e for e in data["events"] if e["target"] == target.to_dict()]
        cutoff = data["deleted"].get(target.digest)
        if any(_time(e["at"]) > at for e in matching) or (
            cutoff is not None and at < _time(cutoff)
        ):
            raise ValueError("删除时间不能倒退")
        data["events"] = [e for e in data["events"] if e["target"] != target.to_dict()]
        data["deleted"][target.digest] = at.isoformat()
        self._write(user_id, data)
        return len(matching)
