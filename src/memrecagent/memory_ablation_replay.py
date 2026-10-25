"""冻结输入/源码后离线重放；结构化oracle不冒充自动记忆或真实用户收益。"""

import json
from datetime import datetime, timedelta
from hashlib import sha256
from math import isfinite
from pathlib import Path

from memrecagent.memory_ablation import run_ranking_ablation
from memrecagent.memory_ablation_metrics import GROUPS, summarize_ablation
from memrecagent.memory_policy import ContextPreference
from memrecagent.memory_product_tool import _read
from memrecagent.memory_products import MemoryProductRecommender, decode_product_entity_fact
from memrecagent.memory_store import JsonlMemoryStore
from memrecagent.product_attributes import decode_request_constraints, decode_request_relaxations
from memrecagent.product_bundle import load_bundle


def _parse_request(value: dict) -> dict:
    required = {"user_id", "session_id", "beneficiary_id", "category", "as_of", "strategy", "k"}
    optional = {
        "requested_entity",
        "constraints",
        "request_id",
        "relaxations",
        "context_preferences",
        "change_scope",
        "soft_preferences",
        "soft_relaxations",
        "use_soft_attributes",
        "soft_weight",
        "max_memory_age_seconds",
    }
    if (
        not isinstance(value, dict)
        or not required <= value.keys()
        or value.keys() - required - optional
    ):
        raise ValueError("请求字段无效；答案不得进入排名请求")
    result = value.copy()
    result["as_of"] = datetime.fromisoformat(result["as_of"])
    result["constraints"] = decode_request_constraints(result.pop("constraints", []))
    result["soft_preferences"] = decode_request_constraints(result.pop("soft_preferences", []))
    result["relaxations"] = decode_request_relaxations(result.pop("relaxations", []))
    result["soft_relaxations"] = decode_request_relaxations(result.pop("soft_relaxations", []))
    contexts = result.pop("context_preferences", [])
    if not isinstance(contexts, list) or len(contexts) > 100:
        raise ValueError("会话线索过多")
    result["context_preferences"] = tuple(
        ContextPreference(**(c | {"available_at": datetime.fromisoformat(c["available_at"])}))
        for c in contexts
    )
    if "max_memory_age_seconds" in result:
        seconds = result.pop("max_memory_age_seconds")
        if type(seconds) not in (int, float) or not isfinite(seconds) or seconds <= 0:
            raise ValueError("时效参数无效")
        result["max_memory_age"] = timedelta(seconds=seconds)
    if result["strategy"] not in ("mf", "bpr", "bpr_cold_popularity"):
        raise ValueError("四组需要已有算法策略")
    if type(result["k"]) is not int or not 1 <= result["k"] <= 50:
        raise ValueError("k必须是1至50整数")
    return result


def _observation(case: dict, groups: dict) -> dict:
    return {
        key: case[key] for key in ("case_id", "cluster_id", "stratum", "relevant", "excluded")
    } | {"groups": groups}


def _hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {name: sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def replay_ablation(
    *,
    bundle: Path,
    facts_path: Path,
    suite_path: Path,
    memory_root: Path,
    output_dir: Path,
    normalisation: str = "minmax",
    run: bool = False,
) -> dict:
    """不调用模型；先验证全部案例，再创建不可覆盖产物；执行失败保留。"""
    _, suite = _read(suite_path, 2 * 1024 * 1024)
    if (
        suite.keys() != {"schema_version", "evidence_kind", "label_source", "cases"}
        or type(suite["schema_version"]) is not int
        or suite["schema_version"] != 1
        or suite["evidence_kind"] != "structured_oracle"
        or not isinstance(suite["label_source"], str)
        or not suite["label_source"].strip()
        or not isinstance(suite["cases"], list)
        or not 1 <= len(suite["cases"]) <= 10000
        or normalisation not in ("minmax", "rank")
    ):
        raise ValueError("套件字段无效")
    requests = []
    placeholders = []
    user_clusters = {}
    for case in suite["cases"]:
        if not isinstance(case, dict) or case.keys() != {
            "case_id",
            "cluster_id",
            "stratum",
            "request",
            "relevant",
            "excluded",
        }:
            raise ValueError("案例字段无效")
        request = _parse_request(case["request"])
        user = request["user_id"]
        if not isinstance(user, str) or not user.strip() or len(user) > 128:
            raise ValueError("用户标识无效")
        cluster = user_clusters.setdefault(user.strip(), case["cluster_id"])
        if cluster != case["cluster_id"]:
            raise ValueError("同一用户不能拆成独立聚类")
        requests.append(request)
        placeholders.append(
            _observation(
                case,
                {
                    g: {"item_ids": [], "status": "error", "algorithm_effective": False}
                    for g in GROUPS
                },
            )
        )
    k = requests[0]["k"]
    if any(r["k"] != k for r in requests):
        raise ValueError("同一套件必须固定k")
    # 预检标签和配对完整性，不向排名器发送这些字段。
    summarize_ablation(placeholders, k=k, n_resamples=1)
    plan = {
        "mode": "plan",
        "n_cases": len(requests),
        "n_groups": 4,
        "api_calls": 0,
        "evidence_kind": "structured_oracle",
        "normalisation": normalisation,
    }
    if not run:
        return plan
    inputs = {"facts.json": facts_path, "suite.json": suite_path}
    inputs.update(
        {f"bundle/{p.relative_to(bundle)}": p for p in sorted(bundle.rglob("*")) if p.is_file()}
    )
    root = Path(__file__).resolve().parents[2]
    sources = {p.name: p for p in sorted(Path(__file__).parent.glob("*.py"))}
    sources["evaluate_ranking_ablation.py"] = root / "scripts/evaluate_ranking_ablation.py"
    sources["pyproject.toml"] = root / "pyproject.toml"
    sources["uv.lock"] = root / "uv.lock"
    # 避免产物写入输入包、源记忆或源码树，也避免输出祖先覆盖来源。
    resolved = output_dir.resolve()
    protected = [bundle.resolve(), memory_root.resolve(), Path(__file__).parent.resolve()]
    if any(resolved == p or resolved in p.parents or p in resolved.parents for p in protected):
        raise ValueError("产物目录不能覆盖输入或源记忆")
    if any(resolved == p.resolve() or resolved in p.resolve().parents for p in inputs.values()):
        raise ValueError("产物目录不能包含源输入")
    before_inputs, before_sources = _hashes(inputs), _hashes(sources)
    service = load_bundle(bundle)
    _, facts_doc = _read(facts_path, 2 * 1024 * 1024)
    if (
        facts_doc.keys() != {"bundle_id", "facts"}
        or facts_doc["bundle_id"] != service.bundle_id
        or not isinstance(facts_doc["facts"], list)
        or len(facts_doc["facts"]) > 5000
    ):
        raise ValueError("实体事实版本无效")
    facts = tuple(decode_product_entity_fact(f) for f in facts_doc["facts"])
    store = JsonlMemoryStore(memory_root)
    guard = MemoryProductRecommender(service, store, facts, facts_bundle_id=service.bundle_id)
    before_memory = {r["user_id"]: guard.state_token(r["user_id"]) for r in requests}
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "source").mkdir()
    for name, path in sources.items():
        (output_dir / "source" / name).write_bytes(path.read_bytes())
    (output_dir / "suite.json").write_bytes(suite_path.read_bytes())
    (output_dir / "facts.json").write_bytes(facts_path.read_bytes())
    manifest = plan | {
        "mode": "offline",
        "bundle_id": service.bundle_id,
        "input_sha256": before_inputs,
        "source_sha256": before_sources,
        "memory_state_tokens": before_memory,
        "label_source": suite["label_source"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    observations = []
    errors = 0
    with (output_dir / "cases.jsonl").open("x") as stream:
        for case, request in zip(suite["cases"], requests, strict=True):
            try:
                arms = run_ranking_ablation(
                    service, store, facts, request=request, normalisation=normalisation
                )
                groups = {
                    g: {
                        "item_ids": [i["item_id"] for i in arm["ranking"]["items"]],
                        "status": arm["ranking"]["status"],
                        "algorithm_effective": arm["algorithm_effective"],
                    }
                    for g, arm in arms.items()
                }
                trace = {"arms": arms}
            except (ValueError, TypeError, KeyError, OSError, FloatingPointError) as error:
                errors += 1
                groups = {
                    g: {"item_ids": [], "status": "error", "algorithm_effective": False}
                    for g in GROUPS
                }
                trace = {"error_type": type(error).__name__}
            observation = _observation(case, groups)
            observations.append(observation)
            stream.write(json.dumps(observation | trace, ensure_ascii=False) + "\n")
            stream.flush()
    unchanged = (
        _hashes(inputs) == before_inputs
        and _hashes(sources) == before_sources
        and before_memory == {u: guard.state_token(u) for u in before_memory}
    )
    report = {
        "evidence_kind": suite["evidence_kind"],
        "label_source": suite["label_source"],
        "api_calls": 0,
        "n_execution_errors": errors,
        "inputs_unchanged": unchanged,
        "summary": summarize_ablation(observations, k=k),
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report
