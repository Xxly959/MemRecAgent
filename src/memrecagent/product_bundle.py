"""冻结并核验本地推理包，服务进程无需访问训练/测试数据目录。"""

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import torch

from memrecagent.bpr_evaluation import FittedBPR
from memrecagent.product_models import load_fitted_model
from memrecagent.product_service import ProductService


def _identity(manifest: dict) -> str:
    value = {key: item for key, item in manifest.items() if key != "bundle_id"}
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return "products-" + sha256(data.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def save_bundle(service: ProductService, output: Path, *, provenance: dict) -> dict:
    """新目录写入完整自包含快照，清单最后发布；不覆盖旧包。"""
    if output.exists():
        raise FileExistsError("推理包目录已经存在，保留原内容")
    if set(service.models) != {"mf", "bpr"} or not 0 < len(service.counts) <= 5000:
        raise ValueError("冻结包需要两种已拟合模型和受控非空候选")
    for model in service.models.values():
        if model.items.keys() != service.counts.keys():
            raise ValueError("模型与候选词表不一致")
    output.mkdir(parents=True, exist_ok=False)
    data = {
        "counts": service.counts,
        "seen": {user: sorted(items) for user, items in service.seen.items()},
        "products": {
            item: {
                "title": product.get("title"),
                "categories": product.get("categories") or [],
                "source": product.get("source"),
                "line_number": product.get("line_number"),
            }
            for item, product in service.products.items()
            if item in service.counts
        },
        "dataset_revision": service.dataset_revision,
        "metadata_url": service.metadata_url,
    }
    _write_json(output / "service.json", data)
    for name, fitted in service.models.items():
        torch.save(
            {
                "model_kind": name,
                "state_dict": fitted.model.state_dict(),
                "users": fitted.users,
                "items": fitted.items,
                "seen": {user: sorted(items) for user, items in fitted.seen.items()},
                "active_users": sorted(fitted.active_users)
                if isinstance(fitted, FittedBPR)
                else None,
                "config": {"factors": fitted.model.user_vectors.embedding_dim},
            },
            output / f"{name}.pt",
        )
    manifest = {
        "schema_version": 1,
        "protocol": "p2-products-v1",
        "as_of_ms": service.as_of_ms,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "provenance": provenance,
        "files": {
            name: sha256((output / name).read_bytes()).hexdigest()
            for name in ["service.json", "mf.pt", "bpr.pt"]
        },
    }
    manifest["bundle_id"] = _identity(manifest)
    _write_json(output / "bundle.json", manifest)
    return manifest


def load_bundle(path: Path) -> ProductService:
    """先核对完整版本与所有文件，再恢复CPU推理对象；不读取原始数据。"""
    manifest = json.loads((path / "bundle.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("bundle_id") != _identity(manifest):
        raise ValueError("推理包版本清单不匹配")
    if set(manifest["files"]) != {"service.json", "mf.pt", "bpr.pt"}:
        raise ValueError("推理包文件集合不符")
    for name, digest in manifest["files"].items():
        if sha256((path / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"{name}: 推理包SHA256不符")
    data = json.loads((path / "service.json").read_text())
    counts = data["counts"]
    if not 0 < len(counts) <= 5000 or any(type(n) is not int or n < 0 for n in counts.values()):
        raise ValueError("推理包候选或热门计数非法")
    models = {name: load_fitted_model(path / f"{name}.pt") for name in ["mf", "bpr"]}
    if isinstance(models["mf"], FittedBPR) or not isinstance(models["bpr"], FittedBPR):
        raise ValueError("推理包模型类型不符")
    if any(model.items.keys() != counts.keys() for model in models.values()):
        raise ValueError("推理包模型词表与候选不符")
    return ProductService(
        bundle_id=manifest["bundle_id"],
        as_of_ms=manifest["as_of_ms"],
        counts=counts,
        seen={user: set(items) for user, items in data["seen"].items()},
        products=data["products"],
        models=models,
        dataset_revision=data["dataset_revision"],
        metadata_url=data["metadata_url"],
    )
