"""回放器的可信乐器环境；真实画像仅在本机绑定，不进入语言消息。"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from pathlib import Path

from memrecagent.entity_aliases import EntityAliases, decode_entity_aliases
from memrecagent.memory import _text, _utc
from memrecagent.memory_extraction import unique_object
from memrecagent.memory_products import ProductEntityFact, decode_product_entity_fact
from memrecagent.product_bundle import load_bundle
from memrecagent.product_service import ProductService


@dataclass(frozen=True)
class NativeReplayEnvironment:
    service: ProductService = field(repr=False)
    facts: tuple[ProductEntityFact, ...] = field(repr=False)
    aliases: EntityAliases = field(repr=False)
    users: dict[str, str] = field(repr=False)
    base_at: datetime
    source_files: dict[str, Path] = field(repr=False)
    source_sha256: dict[str, str] = field(repr=False)
    category: str = "musical_instruments"

    def reference_state(self, facts: list[dict]) -> list[tuple]:
        """只用于评分，不把商品实体伪称品牌，也不补运行状态。"""
        result = []
        for row in facts:
            if (
                row.keys() != {"subject", "category", "target_type", "entity", "relation"}
                or row["subject"] != "self"
                or row["category"] != self.category
                or row["target_type"] != "product"
                or row["entity"] not in self.service.counts
                or row["relation"] not in {"prefer", "dislike", "exclude", "neutral"}
            ):
                raise ValueError("invalid native reference")
            result.append(("self", self.category, "product", row["entity"], row["relation"]))
        return sorted(result)

    def validate_messages(self, messages: list[dict]) -> None:
        payload = json.loads(messages[-1]["content"])

        def walk(value):
            if isinstance(value, dict):
                if {"user_id", "reference", "item_scores", "profile_bindings"} & value.keys():
                    raise ValueError("private or scoring field in runtime payload")
                for item in value.values():
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(payload)
        text = json.dumps(messages, ensure_ascii=False)
        if any(len(user) >= 12 and user in text for user in self.users.values()):
            raise ValueError("native identity in runtime payload")


def load_native_environment(path: Path) -> NativeReplayEnvironment:
    """只加载本机受校验包及显式映射，路径相对配置目录，不读取Key。"""
    path = Path(path).resolve()

    def read_json(file: Path, limit: int) -> dict:
        with file.open("rb") as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("native environment input too large")
        value = json.loads(raw, object_pairs_hook=unique_object)
        if not isinstance(value, dict):
            raise ValueError("native environment input must be an object")
        return value

    config = read_json(path, 65536)
    if config.keys() != {"bundle", "facts", "aliases", "user_bindings", "base_at"}:
        raise ValueError("invalid native environment fields")
    locations = {
        name: (path.parent / _text(config[name], name)).resolve()
        for name in ("bundle", "facts", "aliases", "user_bindings")
    }
    service = load_bundle(locations["bundle"])
    mapping = read_json(locations["facts"], 2 * 1024 * 1024)
    if mapping.keys() != {"bundle_id", "facts"} or mapping["bundle_id"] != service.bundle_id:
        raise ValueError("native catalog mismatch")
    facts = tuple(decode_product_entity_fact(row) for row in mapping["facts"])
    if len(facts) != len(service.counts) or {f.item_id for f in facts} != service.counts.keys():
        raise ValueError("native catalog coverage mismatch")
    with locations["aliases"].open("rb") as stream:
        aliases = decode_entity_aliases(stream.read(2 * 1024 * 1024 + 1))
    if any(
        e.category != "musical_instruments" or e.canonical not in service.counts
        for e in aliases.entries
    ):
        raise ValueError("native aliases mismatch")
    users = read_json(locations["user_bindings"], 65536)
    if not users or len(set(users.values())) != len(users):
        raise ValueError("native users must be distinct")
    for episode, user in users.items():
        if len(_text(episode, "episode")) > 128 or len(_text(user, "user")) > 128:
            raise ValueError("native binding length")
    base_at = _utc(datetime.fromisoformat(config["base_at"]), "base_at")
    if any(f.available_at >= base_at for f in facts):
        raise ValueError("catalog is unavailable at replay start")
    files = {"native/environment.json": path}
    files.update(
        {f"native/{name}.json": locations[name] for name in ("facts", "aliases", "user_bindings")}
    )
    files.update(
        {f"native/bundle/{p.name}": p for p in locations["bundle"].iterdir() if p.is_file()}
    )
    hashes = {name: sha256(p.read_bytes()).hexdigest() for name, p in files.items()}
    return NativeReplayEnvironment(service, facts, aliases, users, base_at, files, hashes)
