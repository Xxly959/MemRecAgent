"""真实已拟合画像的结构化反事实机制审计；不伪称用户偏好或预测效用。"""

import json
from collections import Counter
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path

from memrecagent.bpr_evaluation import FittedBPR
from memrecagent.memory import PreferenceMemory, _text, _utc
from memrecagent.memory_ablation import run_ranking_ablation
from memrecagent.memory_products import ProductEntityFact
from memrecagent.memory_store import JsonlMemoryStore
from memrecagent.product_bundle import load_bundle
from memrecagent.product_service import ProductService


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def select_profiles(service: ProductService, *, count: int) -> tuple[str, ...]:
    if type(count) is not int or not 1 <= count <= 64:
        raise ValueError("活跃样本数须为1至64")
    bpr = service.models.get("bpr")
    mf = service.models.get("mf")
    if not isinstance(bpr, FittedBPR) or mf is None:
        raise ValueError("需要既有两种模型")
    catalog = set(service.counts)
    eligible = [
        u
        for u in bpr.active_users
        if u in bpr.users
        and u in mf.users
        and u in service.seen
        and u == u.strip()
        and len(catalog) > len(catalog & service.seen[u])
    ]
    if len(eligible) < count:
        raise ValueError("真实活跃画像不足，不借用或改名映射")
    return tuple(sorted(eligible, key=lambda u: (_digest("profile-audit-v1:" + u), u))[:count])


def target_item(service: ProductService, user_id: str) -> str:
    candidates = service.counts.keys() - service.seen.get(user_id, set())
    if not candidates:
        raise ValueError("未见候选为空")
    return min(
        candidates, key=lambda item: (_digest("profile-target-v1:" + user_id + ":" + item), item)
    )


def _hashes(files: list[Path]) -> dict[str, str]:
    return {str(p): sha256(p.read_bytes()).hexdigest() for p in files}


def _parameters(service: ProductService) -> str:
    h = sha256()
    for name, model in sorted(service.models.items()):
        for key, tensor in sorted(model.model.state_dict().items()):
            h.update((name + key + str(tensor.shape) + str(tensor.dtype)).encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run_profile_audit(
    *,
    bundle: Path,
    output: Path,
    count: int,
    cold_count: int,
    as_of: datetime,
    category: str,
    run: bool = False,
) -> dict:
    if (
        type(count) is not int
        or not 1 <= count <= 64
        or type(cold_count) is not int
        or not 1 <= cold_count <= 16
    ):
        raise ValueError("样本数量越界")
    _text(category, "category")
    as_of = _utc(as_of, "as_of")
    plan = dict(
        mode="plan",
        evidence_kind="real_profile_structured_counterfactual",
        language_memory_evaluated=False,
        utility_evaluated=False,
        api_calls=0,
        active_count=count,
        cold_count=cold_count,
        conditions=["prefer", "exclude"],
        k=10,
    )
    if not run:
        return plan
    if output.exists():
        raise FileExistsError("保留既有审计产物")
    resolved, source = output.resolve(), Path(__file__).parent.resolve()
    if (
        resolved.is_relative_to(bundle.resolve())
        or bundle.resolve().is_relative_to(resolved)
        or resolved.is_relative_to(source)
        or source.is_relative_to(resolved)
    ):
        raise ValueError("输出不能覆盖输入/源码")
    bundle_files = sorted(p for p in bundle.iterdir() if p.is_file())
    source_files = [
        *sorted(Path(__file__).parent.glob("*.py")),
        Path(__file__).parents[2] / "scripts/audit_warm_profiles.py",
    ]
    before = _hashes(bundle_files + source_files)
    service = load_bundle(bundle)
    fitted_before = _parameters(service)
    manifest = json.loads((bundle / "bundle.json").read_text())
    available = datetime.fromisoformat(manifest["created_at_utc"])
    if available >= as_of:
        raise ValueError("查询须晚于实体身份包可用时刻")
    profiles = select_profiles(service, count=count)
    cold = tuple(f"profile-audit-cold-{i}" for i in range(cold_count))
    known = set(service.seen) | set().union(*(set(m.users) for m in service.models.values()))
    if any(u in known for u in cold):
        raise ValueError("冷控制标识已存在，拒绝暗改用户")
    # 实体是精确目录ID，不是从标题猜出来的品牌，类别由可信调用方声明。
    facts = tuple(
        ProductEntityFact(
            item,
            category,
            "catalog-item:" + item,
            f"bundle:{service.bundle_id}:counts:{item}",
            available,
        )
        for item in sorted(service.counts)
    )
    cases = [
        dict(
            case_id=f"{stratum}-{index}-{condition}",
            user_id=u,
            profile_digest=_digest(u),
            stratum=stratum,
            condition=condition,
            target=target_item(service, u),
        )
        for stratum, users in (("active", profiles), ("cold", cold))
        for index, u in enumerate(users)
        for condition in ("prefer", "exclude")
    ]
    output.mkdir(parents=True, exist_ok=False)
    (output / "source").mkdir()
    for p in source_files:
        (output / "source" / p.name).write_bytes(p.read_bytes())
    # 只有私有计划保存真实owner ID；普通统计/trace只输出digest，不对外发送。
    (output / "private-plan.json").write_text(
        json.dumps(
            dict(cases=cases, as_of=as_of.isoformat(), category=category),
            ensure_ascii=False,
            indent=2,
        )
    )
    (output / "manifest.json").write_text(
        json.dumps(
            plan
            | dict(
                mode="offline",
                bundle_id=service.bundle_id,
                input_and_source_sha256=before,
                profile_selection="sha256 profile-audit-v1; active/known/seen/unseen only",
                target_selection="sha256 profile-target-v1; unseen only; no scores or labels",
                entity_kind="exact_catalog_item_id",
                facts_available_at=available.isoformat(),
                metadata_url=service.metadata_url,
                counterfactual_memory_not_observed_user_statement=True,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    inventory = dict(
        n_catalog_items=len(service.counts),
        n_seen_users=len(service.seen),
        n_mf_users=len(service.models["mf"].users),
        n_bpr_users=len(service.models["bpr"].users),
        n_bpr_active_users=len(service.models["bpr"].active_users),
        metadata_has_brand_field=any("brand" in p for p in service.products.values()),
        metadata_category_roots=dict(
            Counter((p.get("categories") or ["unknown"])[0] for p in service.products.values())
        ),
    )
    summaries = {
        s: dict(
            n_cases=0,
            n_errors=0,
            algorithm_effective_arms=0,
            algorithm_rank_changes=0,
            memory_pop_rank_changes=0,
            memory_bpr_rank_changes=0,
        )
        for s in ("active", "cold")
    }
    observations = []
    with (output / "cases.jsonl").open("x") as stream:
        for case in cases:
            summary = summaries[case["stratum"]]
            summary["n_cases"] += 1
            row = {k: v for k, v in case.items() if k != "user_id"}
            try:
                store = JsonlMemoryStore(output / "counterfactual-memory" / case["condition"])
                store.add(
                    PreferenceMemory(
                        memory_id=case["case_id"],
                        user_id=case["user_id"],
                        category=category,
                        entity="catalog-item:" + case["target"],
                        relation=case["condition"],
                        evidence_ref="structured-counterfactual:" + case["case_id"],
                        evidence_text="结构化反事实控制；不是该真实用户的言语或已观察喜好。",
                        occurred_at=as_of - timedelta(microseconds=1),
                        available_at=as_of - timedelta(microseconds=1),
                    )
                )
                groups = run_ranking_ablation(
                    service,
                    store,
                    facts,
                    request=dict(
                        user_id=case["user_id"],
                        session_id="profile-audit",
                        beneficiary_id="self",
                        category=category,
                        as_of=as_of,
                        strategy="bpr_cold_popularity",
                        k=10,
                    ),
                )
                lists = {
                    g: [i["item_id"] for i in r["ranking"]["items"]] for g, r in groups.items()
                }
                summary["algorithm_effective_arms"] += sum(
                    r["algorithm_effective"] for r in groups.values()
                )
                summary["algorithm_rank_changes"] += lists["G00"] != lists["G01"]
                summary["memory_pop_rank_changes"] += lists["G00"] != lists["G10"]
                summary["memory_bpr_rank_changes"] += lists["G01"] != lists["G11"]
                row.update(status="ok", groups=groups)
            except Exception as error:
                summary["n_errors"] += 1
                row.update(status="error", error_type=type(error).__name__)
            observations.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
    unchanged = before == _hashes(bundle_files + source_files)
    report = plan | dict(
        mode="offline",
        n_cases=len(cases),
        n_execution_errors=sum(s["n_errors"] for s in summaries.values()),
        inventory=inventory,
        by_stratum=summaries,
        inputs_and_source_unchanged=unchanged,
        loaded_parameters_unchanged=fitted_before == _parameters(service),
        target_top10_by_condition={
            condition: {
                g: sum(
                    row.get("status") == "ok"
                    and any(
                        i["item_id"] == row["target"] for i in row["groups"][g]["ranking"]["items"]
                    )
                    for row in observations
                    if row["condition"] == condition
                )
                for g in ("G00", "G01", "G10", "G11")
            }
            for condition in ("prefer", "exclude")
        },
    )
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    )
    return report
