"""可信目录别名表；只做声明的整实体映射，绝不由模型写入。"""

import json
import os
import tempfile
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

ENTITY_ALIASES_FILE = "entity-aliases.json"
MAX_REGISTRY_BYTES = 2 * 1024 * 1024


def _text(value: object, *, limit: int = 128) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError("实体声明字段无效")
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise ValueError("实体声明编码无效") from None
    return value


def _key(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().strip()


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("实体声明字段重复")
        result[key] = value
    return result


@dataclass(frozen=True)
class AliasEntry:
    category: str
    canonical: str
    aliases: tuple[str, ...]
    source: str


@dataclass(frozen=True)
class EntityAliases:
    """不可变表：未知保留原写法，来源与版本摘要可供下游绑定。"""

    entries: tuple[AliasEntry, ...]
    digest: str
    _mapping: Mapping[tuple[str, str], str]
    _sources: Mapping[tuple[str, str], str]

    @classmethod
    def from_document(cls, value: object) -> "EntityAliases":
        if (
            not isinstance(value, dict)
            or value.keys() != {"schema_version", "entries"}
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 1
            or not isinstance(value["entries"], list)
            or not 1 <= len(value["entries"]) <= 5000
        ):
            raise ValueError("实体表字段或版本无效")
        entries = []
        mapping: dict[tuple[str, str], str] = {}
        sources: dict[tuple[str, str], str] = {}
        for row in value["entries"]:
            if not isinstance(row, dict) or row.keys() != {
                "category",
                "canonical",
                "aliases",
                "source",
            }:
                raise ValueError("实体声明字段无效")
            category = _text(row["category"], limit=64)
            canonical = _text(row["canonical"])
            source = _text(row["source"], limit=512)
            raw_aliases = row["aliases"]
            if not isinstance(raw_aliases, list) or len(raw_aliases) > 16:
                raise ValueError("别名列表越界")
            aliases = tuple(sorted({_text(alias, limit=256) for alias in raw_aliases}))
            owner = (category, canonical)
            if owner in sources:
                raise ValueError("规范实体声明重复")
            sources[owner] = source
            for name in (canonical, *aliases):
                key = (category, _key(name))
                if key in mapping and mapping[key] != canonical:
                    raise ValueError("别名指向多个实体，拒绝自动合并")
                mapping[key] = canonical
            entries.append(AliasEntry(category, canonical, aliases, source))
        entries.sort(key=lambda e: (e.category, e.canonical))
        document = dict(
            schema_version=1,
            entries=[
                dict(
                    category=e.category,
                    canonical=e.canonical,
                    aliases=list(e.aliases),
                    source=e.source,
                )
                for e in entries
            ],
        )
        raw = json.dumps(document, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
        if len(raw) > MAX_REGISTRY_BYTES:
            raise ValueError("实体表过大")
        return cls(
            tuple(entries),
            sha256(raw).hexdigest(),
            MappingProxyType(mapping),
            MappingProxyType(sources),
        )

    def canonical(self, category: str, entity: str) -> str:
        _text(category, limit=64)
        # 未知原始实体不受新表的128字符声明上限约束，旧调用方保留其验证边界。
        if not isinstance(entity, str) or not entity.strip():
            raise ValueError("实体无效")
        return self._mapping.get((category, _key(entity)), entity)

    def source_for(self, category: str, entity: str) -> str | None:
        return self._sources.get((category, self.canonical(category, entity)))

    def to_document(self) -> dict:
        return dict(
            schema_version=1,
            entries=[
                dict(
                    category=e.category,
                    canonical=e.canonical,
                    aliases=list(e.aliases),
                    source=e.source,
                )
                for e in self.entries
            ],
        )


def load_entity_aliases(root: Path) -> EntityAliases | None:
    root = Path(root)
    path = root / ENTITY_ALIASES_FILE
    if path.is_symlink() or (root.is_symlink() and path.exists()):
        raise ValueError("实体表不接受符号链接")
    try:
        with path.open("rb") as f:
            raw = f.read(MAX_REGISTRY_BYTES + 1)
    except FileNotFoundError:
        return None
    if len(raw) > MAX_REGISTRY_BYTES:
        raise ValueError("实体表过大")
    return decode_entity_aliases(raw)


def decode_entity_aliases(raw: bytes) -> EntityAliases:
    if not isinstance(raw, bytes) or len(raw) > MAX_REGISTRY_BYTES:
        raise ValueError("实体表过大或类型无效")
    try:
        return EntityAliases.from_document(
            json.loads(raw.decode("utf-8"), object_pairs_hook=_unique)
        )
    except (UnicodeError, ValueError, TypeError):
        raise ValueError("实体表无效") from None


def _empty_namespace(root: Path) -> None:
    if root.is_symlink() or (root / "feedback").is_symlink() or (root / ".management").is_symlink():
        raise ValueError("命名空间目录无效")
    pending = root / ".management"
    if (
        any(root.glob("*.jsonl"))
        or any((root / "feedback").glob("*.json"))
        or (pending.exists() and any(pending.iterdir()))
    ):
        raise ValueError("已有数据需要显式迁移，不安装新实体表")


def install_entity_aliases(root: Path, document: object) -> EntityAliases:
    """仅新空namespace可安装；同目录原子发布，不提供覆盖/模型写入口。"""
    table = EntityAliases.from_document(document)
    root = Path(root)
    path = root / ENTITY_ALIASES_FILE
    if path.exists() or path.is_symlink():
        raise FileExistsError("保留已有实体表")
    _empty_namespace(root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=root, prefix=".entities-", delete=False
        ) as f:
            temporary = Path(f.name)
            json.dump(table.to_document(), f, ensure_ascii=False, sort_keys=True, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        _empty_namespace(root)
        if path.exists() or path.is_symlink():
            raise FileExistsError("实体表已存在")
        os.replace(temporary, path)
        fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return table


def checked_entity_aliases(root: Path, expected_digest: str | None) -> EntityAliases | None:
    """存储实例固定表版本；即使尚无用户记录，也不接纳中途换表。"""
    table = load_entity_aliases(root)
    if (table.digest if table is not None else None) != expected_digest:
        raise ValueError("entity_registry_changed")
    return table
