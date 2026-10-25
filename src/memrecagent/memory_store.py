"""本人偏好JSONL存储；单进程串行写入，修订保留原始证据。"""

import json
import os
import tempfile
from dataclasses import replace
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Literal
from uuid import uuid4

from memrecagent.attribute_memory import AttributeMemory
from memrecagent.entity_aliases import EntityAliases, checked_entity_aliases, load_entity_aliases
from memrecagent.memory import PreferenceMemory, _memory_age, _text, _utc
from memrecagent.memory_management_guard import ensure_no_pending
from memrecagent.product_attributes import NUMERIC_UNITS, UNITS

type MemoryRecord = PreferenceMemory | AttributeMemory


def _history_key(memory: MemoryRecord) -> tuple:
    if isinstance(memory, AttributeMemory):
        return ("attribute", memory.category, memory.attribute, memory.operator, memory.is_hard)
    return ("brand", memory.category, memory.entity)


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("JSON字段重复")
        value[key] = item
    return value


def _validate_history(records: list[MemoryRecord]) -> None:
    known: dict[str, MemoryRecord] = {}
    heads: dict[tuple, str] = {}
    for memory in records:
        key = _history_key(memory)
        if memory.memory_id in known:
            raise ValueError("重复记忆ID")
        if memory.supersedes is None:
            if key in heads:
                raise ValueError("已有偏好需要显式修订")
        else:
            parent = known.get(memory.supersedes)
            if parent is None or heads.get(key) != memory.supersedes:
                raise ValueError("修订引用不存在、跨对象或不是当前版本")
            if (memory.user_id, memory.subject_id) != (parent.user_id, parent.subject_id):
                raise ValueError("修订归属不一致")
            if (
                memory.occurred_at <= parent.occurred_at
                or memory.available_at < parent.available_at
            ):
                raise ValueError("修订时间不能倒退或覆盖同一时刻")
        known[memory.memory_id] = memory
        heads[key] = memory.memory_id


def _scope_digest(category: str, entity: str) -> str:
    """删除标记只存作用域摘要，不保存品牌原文和证据。"""
    _text(category, "category")
    _text(entity, "entity")
    return sha256(json.dumps([category, entity], ensure_ascii=False).encode()).hexdigest()


def _attribute_scope_digest(category: str, attribute: str, operator: str, is_hard: bool) -> str:
    _text(category, "category")
    if not isinstance(attribute, str) or attribute not in UNITS:
        raise ValueError("属性不支持")
    # 长度及类型均与旧品牌摘要的二元素列表隔离。
    return sha256(
        json.dumps(
            ["attribute", category, attribute, operator, is_hard], ensure_ascii=False
        ).encode()
    ).hexdigest()


def _blocked(memory: MemoryRecord, deleted: dict[str, datetime]) -> bool:
    scope = (
        _attribute_scope_digest(memory.category, memory.attribute, memory.operator, memory.is_hard)
        if isinstance(memory, AttributeMemory)
        else _scope_digest(memory.category, memory.entity)
    )
    cutoff = deleted.get(scope)
    return cutoff is not None and (memory.occurred_at <= cutoff or memory.available_at <= cutoff)


def _snapshot(
    records: list[MemoryRecord], deleted: dict[str, datetime], registry_digest: str | None = None
) -> str:
    """只返回摘要；偏好和删除水位均参与版本，空历史不等于未发生删除。"""
    state = {
        "records": [m.to_dict() for m in records],
        "deleted": {key: value.isoformat() for key, value in deleted.items()},
    }
    if registry_digest is not None:
        state["entity_registry_digest"] = registry_digest
    return sha256(json.dumps(state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class JsonlMemoryStore:
    """只持久化本人偏好；有效版本先按查询时刻筛选，再解析修订链。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        table = load_entity_aliases(self.root)
        self.entity_registry_digest = table.digest if table is not None else None

    def entity_aliases(self) -> EntityAliases | None:
        """返回当前仍与本实例固定版本一致的可信表。"""
        return checked_entity_aliases(self.root, self.entity_registry_digest)

    def canonical_entity(self, category: str, entity: str) -> str:
        table = self.entity_aliases()
        _text(entity, "entity")
        return table.canonical(category, entity) if table is not None else entity

    def _path(self, user_id: str) -> Path:
        _text(user_id, "user_id")
        if len(user_id) > 128:
            raise ValueError("user_id: 长度超过128字符")
        return self.root / (sha256(user_id.encode("utf-8")).hexdigest() + ".jsonl")

    def _load(self, user_id: str) -> tuple[list[MemoryRecord], dict[str, datetime]]:
        ensure_no_pending(self.root, user_id)
        table = self.entity_aliases()
        registry_bound = False
        path = self._path(user_id)
        if path.is_symlink():
            raise ValueError(f"{path.name}: 不接受符号链接记忆文件")
        records = []
        deleted: dict[str, datetime] = {}
        try:
            with path.open("rb") as stream:
                for line_number, line in enumerate(stream, 1):
                    try:
                        data = json.loads(line.decode("utf-8"), object_pairs_hook=_unique_keys)
                        if isinstance(data, dict) and data.get("record_type") == "entity_registry":
                            if (
                                data.keys()
                                != {
                                    "record_type",
                                    "schema_version",
                                    "registry_digest",
                                    "owner_digest",
                                }
                                or type(data["schema_version"]) is not int
                                or data["schema_version"] != 1
                                or line_number != 1
                                or registry_bound
                                or table is None
                                or data["registry_digest"] != table.digest
                                or data["owner_digest"] != path.stem
                            ):
                                raise ValueError("entity_registry_binding")
                            registry_bound = True
                            continue
                        if isinstance(data, dict) and data.get("record_type") == "deletion":
                            if data.keys() != {
                                "schema_version",
                                "record_type",
                                "owner_digest",
                                "scope_digest",
                                "deleted_at",
                            }:
                                raise ValueError("删除标记字段无效")
                            scope = data["scope_digest"]
                            if (
                                type(data["schema_version"]) is not int
                                or data["schema_version"] != 1
                                or data["owner_digest"] != path.stem
                                or not isinstance(scope, str)
                                or len(scope) != 64
                                or any(c not in "0123456789abcdef" for c in scope)
                                or scope in deleted
                                or not isinstance(data["deleted_at"], str)
                            ):
                                raise ValueError("删除标记无效")
                            deleted[scope] = _utc(
                                datetime.fromisoformat(data["deleted_at"]), "deleted_at"
                            )
                            continue
                        memory = (
                            AttributeMemory.from_dict(data)
                            if isinstance(data, dict) and data.get("record_type") == "attribute"
                            else PreferenceMemory.from_dict(data)
                        )
                        if (
                            isinstance(memory, PreferenceMemory)
                            and table is not None
                            and table.canonical(memory.category, memory.entity) != memory.entity
                        ):
                            raise ValueError("canonical_entity_required")
                        if memory.user_id != user_id:
                            raise ValueError("用户归属不一致")
                        records.append(memory)
                    except (ValueError, UnicodeError):
                        raise ValueError(f"{path.name}:{line_number}: 记忆记录无效") from None
        except FileNotFoundError:
            return [], {}
        except OSError:
            raise OSError(f"{path.name}: 记忆读取失败") from None
        if table is not None and not registry_bound:
            raise ValueError("entity_namespace_unbound")
        try:
            _validate_history(records)
            if any(_blocked(row, deleted) for row in records):
                raise ValueError("已删除的旧证据不能恢复")
        except ValueError:
            raise ValueError(f"{path.name}: 修订链无效") from None
        return records, deleted

    def add(self, memory: MemoryRecord) -> None:
        """校验旧文件和新修订后原子发布，失败不覆盖旧文件。"""
        self.add_many((memory,))

    def snapshot_token(self, user_id: str) -> str:
        """完整校验后取得当前用户状态摘要；只用于本地串行提交防过期。"""
        return _snapshot(*self._load(user_id), registry_digest=self.entity_registry_digest)

    def add_many(
        self, memories: tuple[MemoryRecord, ...], *, expected_token: str | None = None
    ) -> None:
        """同轮多条先全部校验，再一次发布；任何错误均不留下半批记录。"""
        if not memories or any(
            not isinstance(m, (PreferenceMemory, AttributeMemory)) for m in memories
        ):
            raise ValueError("只接受本人长期偏好记录")
        user_id = memories[0].user_id
        if any(m.user_id != user_id for m in memories):
            raise ValueError("批量记录用户归属不一致")
        aliases = self.entity_aliases()
        memories = tuple(
            replace(m, entity=aliases.canonical(m.category, m.entity))
            if isinstance(m, PreferenceMemory) and aliases is not None
            else m
            for m in memories
        )
        records, deleted = self._load(user_id)
        if expected_token is not None and expected_token != _snapshot(
            records, deleted, self.entity_registry_digest
        ):
            raise ValueError("记忆状态已变化，请重新生成提案")
        if any(_blocked(m, deleted) for m in memories):
            raise ValueError("已删除的旧证据不能再次写入")
        records.extend(memories)
        _validate_history(records)
        self._write(user_id, records, deleted)

    def _write(
        self, user_id: str, records: list[MemoryRecord], deleted: dict[str, datetime]
    ) -> None:
        """偏好和删除水位使用同一个原子发布点，避免双文件中间状态。"""
        table = self.entity_aliases()
        path = self._path(user_id)
        temporary = None
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.root,
                prefix=".memory-",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                if table is not None:
                    stream.write(
                        json.dumps(
                            dict(
                                record_type="entity_registry",
                                schema_version=1,
                                registry_digest=table.digest,
                                owner_digest=path.stem,
                            ),
                            sort_keys=True,
                        )
                        + "\n"
                    )
                for row in records:
                    stream.write(json.dumps(row.to_dict(), ensure_ascii=False, allow_nan=False))
                    stream.write("\n")
                for scope, cutoff in sorted(deleted.items()):
                    stream.write(
                        json.dumps(
                            {
                                "schema_version": 1,
                                "record_type": "deletion",
                                "owner_digest": path.stem,
                                "scope_digest": scope,
                                "deleted_at": cutoff.isoformat(),
                            }
                        )
                    )
                    stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except OSError:
            raise OSError(f"{path.name}: 记忆写入失败") from None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def correct(
        self,
        user_id: str,
        *,
        category: str,
        entity: str,
        expected_head: str,
        relation: Literal["prefer", "dislike", "exclude", "neutral"],
        evidence_ref: str,
        evidence_text: str,
        occurred_at: datetime,
        available_at: datetime,
    ) -> PreferenceMemory:
        """明确纠正现有偏好；校验用户看到的版本，追加新证据而不覆盖旧快照。"""
        entity = self.canonical_entity(category, entity)
        _scope_digest(category, entity)
        _text(expected_head, "expected_head")
        records, _ = self._load(user_id)
        matching = [
            m
            for m in records
            if isinstance(m, PreferenceMemory) and (m.category, m.entity) == (category, entity)
        ]
        if not matching or matching[-1].memory_id != expected_head:
            raise ValueError("待纠正记录不存在或版本已变化，请重新查看")
        memory = PreferenceMemory(
            "memory:" + uuid4().hex,
            user_id,
            category,
            entity,
            relation,
            evidence_ref,
            evidence_text,
            occurred_at,
            available_at,
            supersedes=expected_head,
        )
        self.add(memory)
        return memory

    def delete_entity(
        self,
        user_id: str,
        *,
        category: str,
        entity: str,
        deleted_at: datetime,
    ) -> int:
        """物理移除指定对象的全部历史，并阻止删除时刻及之前的证据重放。"""
        entity = self.canonical_entity(category, entity)
        scope = _scope_digest(category, entity)
        deleted_at = _utc(deleted_at, "deleted_at")
        records, deleted = self._load(user_id)
        matching = [
            m
            for m in records
            if isinstance(m, PreferenceMemory) and (m.category, m.entity) == (category, entity)
        ]
        if any(max(m.occurred_at, m.available_at) > deleted_at for m in matching) or (
            scope in deleted and deleted_at < deleted[scope]
        ):
            raise ValueError("删除时间不能早于已知证据或上次删除")
        remaining = [m for m in records if m not in matching]
        deleted[scope] = deleted_at
        self._write(user_id, remaining, deleted)
        return len(matching)

    def evidence_is_deleted(
        self,
        user_id: str,
        *,
        category: str,
        entity: str,
        available_at: datetime,
        occurred_at: datetime | None = None,
    ) -> bool:
        """调用方在使用旧提案/上下文前核验水位，不向外暴露删除摘要。"""
        entity = self.canonical_entity(category, entity)
        scope = _scope_digest(category, entity)
        available_at = _utc(available_at, "available_at")
        occurred_at = available_at if occurred_at is None else _utc(occurred_at, "occurred_at")
        _, deleted = self._load(user_id)
        cutoff = deleted.get(scope)
        return cutoff is not None and min(occurred_at, available_at) <= cutoff

    def attribute_evidence_is_deleted(
        self, user_id: str, *, category: str, attribute: str, available_at: datetime
    ) -> bool:
        """本次属性也要防旧证据重放；不按硬软/比较方式绕过用户删除。"""
        _text(category, "category")
        if attribute not in UNITS:
            raise ValueError("属性无效")
        available_at = _utc(available_at, "available_at")
        _, deleted = self._load(user_id)
        operators = ("<=", ">=", "==") if attribute in NUMERIC_UNITS else ("==",)
        return any(
            available_at <= deleted[key]
            for op in operators
            for hard in (False, True)
            if (key := _attribute_scope_digest(category, attribute, op, hard)) in deleted
        )

    def list_memories(
        self,
        user_id: str,
        *,
        category: str | None = None,
        entity: str | None = None,
        as_of: datetime,
    ) -> tuple[PreferenceMemory, ...]:
        """返回该时刻已发生且已可用的完整本人证据历史，不隐藏损坏行。"""
        as_of = _utc(as_of, "as_of")
        if category is not None:
            _text(category, "category")
        if entity is not None:
            entity = self.canonical_entity(category, entity) if category is not None else entity
            _text(entity, "entity")
        records, _ = self._load(user_id)
        return tuple(
            sorted(
                (
                    memory
                    for memory in records
                    if isinstance(memory, PreferenceMemory)
                    and (category is None or memory.category == category)
                    and (
                        entity is None
                        or memory.entity == self.canonical_entity(memory.category, entity)
                    )
                    and memory.occurred_at < as_of
                    and memory.available_at < as_of
                ),
                key=lambda memory: (memory.occurred_at, memory.memory_id),
            )
        )

    def active_memories(
        self,
        user_id: str,
        *,
        category: str | None = None,
        entity: str | None = None,
        as_of: datetime,
        max_memory_age: timedelta | None = None,
    ) -> tuple[PreferenceMemory, ...]:
        """晚到修订只在可用后生效；neutral表示撤销态度而非转成喜欢。"""
        max_memory_age = _memory_age(max_memory_age)
        as_of = _utc(as_of, "as_of")
        visible = self.list_memories(user_id, category=category, entity=entity, as_of=as_of)
        replaced = {memory.supersedes for memory in visible if memory.supersedes is not None}
        return tuple(
            memory
            for memory in visible
            if memory.memory_id not in replaced
            and (max_memory_age is None or as_of - memory.occurred_at < max_memory_age)
        )

    def list_attribute_memories(
        self,
        user_id: str,
        *,
        category: str | None = None,
        attribute: str | None = None,
        as_of: datetime,
    ) -> tuple[AttributeMemory, ...]:
        """读取类型化属性历史；仍完整校验混合文件，不能跳过坏品牌行。"""
        as_of = _utc(as_of, "as_of")
        if category is not None:
            _text(category, "category")
        if attribute is not None and (not isinstance(attribute, str) or attribute not in UNITS):
            raise ValueError("属性不支持")
        records, _ = self._load(user_id)
        return tuple(
            sorted(
                (
                    m
                    for m in records
                    if isinstance(m, AttributeMemory)
                    and (category is None or m.category == category)
                    and (attribute is None or m.attribute == attribute)
                    and m.occurred_at < as_of
                    and m.available_at < as_of
                ),
                key=lambda m: (m.occurred_at, m.memory_id),
            )
        )

    def active_attribute_memories(
        self,
        user_id: str,
        *,
        category: str | None = None,
        attribute: str | None = None,
        as_of: datetime,
        max_memory_age: timedelta | None = None,
        include_inactive: bool = False,
    ) -> tuple[AttributeMemory, ...]:
        """先解析可见修订链，再过滤撤销和过期，避免旧预算复活。"""
        if type(include_inactive) is not bool:
            raise ValueError("include_inactive需要布尔值")
        as_of = _utc(as_of, "as_of")
        max_memory_age = _memory_age(max_memory_age)
        visible = self.list_attribute_memories(
            user_id, category=category, attribute=attribute, as_of=as_of
        )
        replaced = {m.supersedes for m in visible if m.supersedes is not None}
        return tuple(
            m
            for m in visible
            if m.memory_id not in replaced
            and (include_inactive or m.active)
            and (max_memory_age is None or as_of - m.occurred_at < max_memory_age)
        )

    def correct_attribute(
        self,
        user_id: str,
        *,
        category: str,
        attribute: str,
        operator: str,
        is_hard: bool,
        expected_head: str,
        value: float | str | bool,
        unit: str,
        active: bool,
        evidence_ref: str,
        evidence_text: str,
        occurred_at: datetime,
        available_at: datetime,
    ) -> AttributeMemory:
        """可信宿主显式纠正；不经此接口推断口头话语的保存意图。"""
        _text(expected_head, "expected_head")
        memory = AttributeMemory(
            "attribute:" + uuid4().hex,
            user_id,
            category,
            attribute,
            operator,
            value,
            unit,
            is_hard,
            active,
            evidence_ref,
            evidence_text,
            occurred_at,
            available_at,
            supersedes=expected_head,
        )
        records, _ = self._load(user_id)
        matching = [m for m in records if _history_key(m) == _history_key(memory)]
        if not matching or matching[-1].memory_id != expected_head:
            raise ValueError("待纠正属性不存在或版本已变化，请重新查看")
        self.add(memory)
        return memory

    def delete_attribute(
        self,
        user_id: str,
        *,
        category: str,
        attribute: str,
        deleted_at: datetime,
    ) -> int:
        """物理删除该属性的全部硬软历史，并阻止换比较方式重放旧证据。"""
        operators = ("<=", ">=", "==") if attribute in NUMERIC_UNITS else ("==",)
        scopes = {
            _attribute_scope_digest(category, attribute, op, hard)
            for op in operators
            for hard in (False, True)
        }
        deleted_at = _utc(deleted_at, "deleted_at")
        records, deleted = self._load(user_id)
        matching = [
            m
            for m in records
            if isinstance(m, AttributeMemory) and (m.category, m.attribute) == (category, attribute)
        ]
        if any(max(m.occurred_at, m.available_at) > deleted_at for m in matching) or any(
            deleted_at < deleted[s] for s in scopes if s in deleted
        ):
            raise ValueError("删除时间不能早于已知证据或上次删除")
        remaining = [m for m in records if m not in matching]
        deleted.update({s: deleted_at for s in scopes})
        self._write(user_id, remaining, deleted)
        return len(matching)
