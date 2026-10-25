"""宿主绑定路径的只读商品工具；模型不能选择用户、文件或修改记忆。"""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timedelta
from hashlib import sha256
from math import isfinite
from pathlib import Path

from memrecagent.memory_feedback_store import JsonFeedbackStore
from memrecagent.memory_policy import ContextPreference
from memrecagent.memory_products import MemoryProductRecommender, decode_product_entity_fact
from memrecagent.memory_store import JsonlMemoryStore
from memrecagent.memory_use_policy import JsonMemoryUseStore
from memrecagent.product_attributes import decode_request_constraints, decode_request_relaxations
from memrecagent.product_bundle import load_bundle


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("重复字段")
        result[key] = value
    return result


def _number(text):
    value = float(text)
    if not isfinite(value):
        raise ValueError("非有限数")
    return value


def _read(path: Path, limit: int) -> tuple[bytes, dict]:
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("文件过大")
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_unique,
        parse_float=_number,
        parse_constant=lambda _: _number("nan"),
    )
    if not isinstance(value, dict):
        raise ValueError("需要对象")
    return raw, value


def _digest_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token(
    bundle: Path, store: JsonlMemoryStore, user_id: str, request_path: Path, facts_path: Path
) -> str:
    policy, revision = JsonMemoryUseStore(store.root / "use-policy").snapshot(user_id)
    state = [
        store.snapshot_token(user_id),
        JsonFeedbackStore(store.root / "feedback").snapshot_token(user_id),
        asdict(policy),
        revision,
        _digest_file(request_path),
        _digest_file(facts_path),
        {
            name: _digest_file(bundle / name)
            for name in ("bundle.json", "service.json", "mf.pt", "bpr.pt")
        },
    ]
    return sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def execute(
    *,
    bundle: Path,
    memory_root: Path,
    facts_path: Path,
    request_path: Path,
    check_token: str | None = None,
) -> dict:
    """绑定配置属于可信宿主；原始证据不在商品回执中回传。"""
    raw_request, request = _read(request_path, 65536)
    required = {"user_id", "session_id", "beneficiary_id", "category", "as_of", "strategy"}
    optional = {
        "k",
        "requested_entity",
        "context_preferences",
        "change_scope",
        "soft_weight",
        "max_memory_age_seconds",
        "normalisation",
        "use_long_term_memory",
        "constraints",
        "request_id",
        "relaxations",
        "soft_preferences",
        "soft_relaxations",
        "use_soft_attributes",
    }
    if not required <= request.keys() or request.keys() - required - optional:
        raise ValueError("请求字段不匹配")
    store = JsonlMemoryStore(memory_root)
    before = _token(bundle, store, request["user_id"], request_path, facts_path)
    if check_token is not None:
        if before != check_token:
            raise ValueError("快照过期")
        return {"valid": True}
    raw_facts, facts = _read(facts_path, 2 * 1024 * 1024)
    if (
        set(facts) != {"bundle_id", "facts"}
        or not isinstance(facts["facts"], list)
        or not 0 < len(facts["facts"]) <= 5000
    ):
        raise ValueError("映射字段不匹配")
    rows = tuple(decode_product_entity_fact(row) for row in facts["facts"])
    request["constraints"] = decode_request_constraints(request.pop("constraints", []))
    request["soft_preferences"] = decode_request_constraints(request.pop("soft_preferences", []))
    request["relaxations"] = decode_request_relaxations(request.pop("relaxations", []))
    request["soft_relaxations"] = decode_request_relaxations(request.pop("soft_relaxations", []))
    contexts = request.pop("context_preferences", [])
    if not isinstance(contexts, list) or len(contexts) > 100:
        raise ValueError("会话线索过多")
    request["context_preferences"] = tuple(
        ContextPreference(**(row | {"available_at": datetime.fromisoformat(row["available_at"])}))
        for row in contexts
    )
    request["as_of"] = datetime.fromisoformat(request["as_of"])
    if "max_memory_age_seconds" in request:
        seconds = request.pop("max_memory_age_seconds")
        if type(seconds) not in (int, float) or not isfinite(seconds) or seconds <= 0:
            raise ValueError("时效无效")
        request["max_memory_age"] = timedelta(seconds=seconds)
    recommender = MemoryProductRecommender(
        load_bundle(bundle), store, rows, facts_bundle_id=facts["bundle_id"]
    )
    result = recommender.recommend(**request)
    # 包括读取前后变化；字节快照也必须是实际解析的配置。
    if (
        _token(bundle, store, request["user_id"], request_path, facts_path) != before
        or _digest_file(request_path) != sha256(raw_request).hexdigest()
        or _digest_file(facts_path) != sha256(raw_facts).hexdigest()
    ):
        raise ValueError("请求状态已改变")
    return result | {"delivery_token": before}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("bundle", "memory-root", "facts", "request"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    parser.add_argument("--check-token")
    args = parser.parse_args()
    try:
        value = execute(
            bundle=args.bundle,
            memory_root=args.memory_root,
            facts_path=args.facts,
            request_path=args.request,
            check_token=args.check_token,
        )
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False)
        if len(raw.encode()) > 65536:
            raise ValueError("回执过大")
    except (ValueError, TypeError, KeyError, OSError, OverflowError, RecursionError, RuntimeError):
        sys.stderr.write("recommendation_unavailable\n")
        return 2
    print(raw, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
