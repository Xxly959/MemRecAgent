"""可信本机跨偏好/反馈删除；双快照预检、阻读标记、可恢复串行发布。"""

import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from memrecagent.entity_aliases import ENTITY_ALIASES_FILE
from memrecagent.memory import _text, _utc
from memrecagent.memory_feedback import FeedbackTarget, _active
from memrecagent.memory_feedback_store import JsonFeedbackStore
from memrecagent.memory_management_guard import ensure_no_pending, management_marker
from memrecagent.memory_store import JsonlMemoryStore, _unique_keys
from memrecagent.product_attributes import UNITS


def _hash(path: Path) -> str | None:
    if path.is_symlink():
        raise ValueError("不接受管理文件符号链接")
    try:
        return sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def _sync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class MemoryManager:
    """调用方负责本地用户身份；不接收LLM写入，也不保证多进程事务。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.memories = JsonlMemoryStore(self.root)
        self.feedback = JsonFeedbackStore(self.root / "feedback")

    def list_feedback(self, user_id: str, *, as_of: datetime) -> list[dict]:
        as_of = _utc(as_of, "as_of")
        events = self.feedback._load(user_id)["events"]
        targets = {FeedbackTarget(**e["target"]) for e in events}
        result = []
        for target in sorted(targets, key=lambda t: (t.kind, t.category, t.target_id)):
            if _active(events, target, as_of):
                latest = next(
                    e
                    for e in reversed(events)
                    if e["target"] == target.to_dict()
                    and datetime.fromisoformat(e["at"]) < as_of
                    and e["action"] == "block"
                )
                result.append(
                    dict(target=target.to_dict(), event_digest=latest["event"], at=latest["at"])
                )
        return result

    def inspect(self, user_id: str, *, as_of: datetime) -> dict:
        """用户显式查看全部类型；不作为模型上下文或自动日志输出。"""
        before = (self.memories.snapshot_token(user_id), self.feedback.snapshot_token(user_id))
        result = dict(
            preferences=[m.to_dict() for m in self.memories.list_memories(user_id, as_of=as_of)],
            attributes=[
                m.to_dict() for m in self.memories.list_attribute_memories(user_id, as_of=as_of)
            ],
            feedback=self.list_feedback(user_id, as_of=as_of),
        )
        if before != (self.memories.snapshot_token(user_id), self.feedback.snapshot_token(user_id)):
            raise ValueError("管理读取期间状态变化")
        return result

    def forget(
        self,
        user_id: str,
        *,
        kind: str,
        category: str,
        target: str,
        at: datetime,
        feedback_only: bool = False,
    ) -> dict:
        if (
            kind not in ("brand", "product", "attribute")
            or type(feedback_only) is not bool
            or (feedback_only and kind == "attribute")
        ):
            raise ValueError("删除类型无效")
        _text(category, "category")
        _text(target, "target")
        if kind == "attribute" and target not in UNITS:
            raise ValueError("属性无效")
        if kind == "brand":
            target = self.memories.canonical_entity(category, target)
        at = _utc(at, "at")
        ensure_no_pending(self.root, user_id)
        memory_path, feedback_path = self.memories._path(user_id), self.feedback._path(user_id)
        registry_path = self.root / ENTITY_ALIASES_FILE
        before_registry = _hash(registry_path)
        before = (_hash(memory_path), _hash(feedback_path))
        records, deleted = self.memories._load(user_id)
        feedback = self.feedback._load(user_id)
        if feedback_path.parent.is_symlink():
            raise ValueError("不接受反馈目录符号链接")
        marker = management_marker(self.root, user_id)
        marker.parent.mkdir(parents=True, exist_ok=True)
        _sync(self.root)
        owner = marker.stem
        stage = marker.parent / (owner + "." + uuid4().hex)
        stage.mkdir()
        try:
            if before_registry is not None:
                raw_registry = registry_path.read_bytes()
                (stage / "memory").mkdir()
                (stage / "memory" / ENTITY_ALIASES_FILE).write_bytes(raw_registry)
                (stage / ENTITY_ALIASES_FILE).write_bytes(raw_registry)
            candidate_memory = JsonlMemoryStore(stage / "memory")
            candidate_feedback = JsonFeedbackStore(stage / "feedback")
            candidate_memory._write(user_id, records, deleted)
            candidate_feedback._write(user_id, feedback)
            count = 0
            if kind == "brand" and not feedback_only:
                count = candidate_memory.delete_entity(
                    user_id, category=category, entity=target, deleted_at=at
                )
            elif kind == "attribute":
                count = candidate_memory.delete_attribute(
                    user_id, category=category, attribute=target, deleted_at=at
                )
            removed = (
                0
                if kind == "attribute"
                else candidate_feedback.delete(user_id, FeedbackTarget(kind, category, target), at)
            )
            # 两个新快照都通过正式严格decoder再发布意图；不留半个已接受的新状态。
            candidate_memory._load(user_id)
            candidate_feedback._load(user_id)
            _sync(candidate_memory.root)
            _sync(candidate_feedback.root)
            _sync(stage)
            if before != (_hash(memory_path), _hash(feedback_path)) or before_registry != _hash(
                registry_path
            ):
                raise ValueError("管理准备期间源状态漂移")
            journal = dict(
                schema_version=1,
                owner_digest=owner,
                stage=stage.name,
                memory_before=before[0],
                feedback_before=before[1],
                memory_after=_hash(candidate_memory._path(user_id)),
                feedback_after=_hash(candidate_feedback._path(user_id)),
            )
            if before_registry is not None:
                journal["registry_sha256"] = before_registry
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=marker.parent, prefix=".intent-", delete=False
            ) as f:
                intent = Path(f.name)
                json.dump(journal, f, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.replace(intent, marker)
                _sync(marker.parent)
            finally:
                intent.unlink(missing_ok=True)
            self.recover(user_id)
            return dict(deleted_memories=count, deleted_feedback=removed)
        finally:
            if not marker.exists() and not stage.is_symlink():
                shutil.rmtree(stage, ignore_errors=True)

    def recover(self, user_id: str) -> bool:
        """显式完成已有删除意图；未知漂移/损坏只阻读，不猜测、不改坏文件。"""
        marker = management_marker(self.root, user_id)
        if marker.parent.is_symlink() or marker.is_symlink():
            raise ValueError("管理标记无效")
        try:
            if marker.stat().st_size > 2048:
                raise ValueError("管理标记无效")
            journal = json.loads(marker.read_text(), object_pairs_hook=_unique_keys)
        except FileNotFoundError:
            return False
        fields = {
            "schema_version",
            "owner_digest",
            "stage",
            "memory_before",
            "feedback_before",
            "memory_after",
            "feedback_after",
        }
        if (
            not isinstance(journal, dict)
            or (journal.keys() != fields and journal.keys() != fields | {"registry_sha256"})
            or type(journal["schema_version"]) is not int
            or journal["schema_version"] != 1
            or journal["owner_digest"] != marker.stem
            or not isinstance(journal["stage"], str)
            or re.fullmatch(re.escape(marker.stem) + r"\.[0-9a-f]{32}", journal["stage"]) is None
        ):
            raise ValueError("管理标记无效")
        for key in ("memory_before", "feedback_before", "memory_after", "feedback_after"):
            value = journal[key]
            if value is None and key.endswith("before"):
                continue
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("管理摘要无效")
        registry_hash = journal.get("registry_sha256")
        if registry_hash is not None and (
            not isinstance(registry_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", registry_hash) is None
        ):
            raise ValueError("管理实体表摘要无效")
        if _hash(self.root / ENTITY_ALIASES_FILE) != registry_hash:
            raise ValueError("管理实体表变化，保留阻读标记")
        stage = marker.parent / journal["stage"]
        if stage.is_symlink() or any(
            (stage / name).is_symlink() for name in ("memory", "feedback")
        ):
            raise ValueError("管理临时目录无效")
        pairs = (
            ("memory", self.memories._path(user_id)),
            ("feedback", self.feedback._path(user_id)),
        )
        # 所有路径先验证，再执行任何一个替换。
        pending = []
        for name, path in pairs:
            if path.parent.is_symlink():
                raise ValueError("管理目标目录无效")
            current = _hash(path)
            if current == journal[name + "_after"]:
                continue
            source = stage / name / path.name
            if current != journal[name + "_before"] or _hash(source) != journal[name + "_after"]:
                raise ValueError("管理状态漂移，保留阻读标记")
            pending.append((source, path))
        for source, path in pending:
            path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, path)
            _sync(path.parent)
        if any(_hash(path) != journal[name + "_after"] for name, path in pairs):
            raise ValueError("管理发布未完成")
        shutil.rmtree(stage, ignore_errors=False) if stage.exists() else None
        marker.unlink()
        _sync(marker.parent)
        return True
