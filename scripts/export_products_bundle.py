"""从同版本数据与已保存权重生成本地冻结推理包，不重新训练。"""

import argparse
import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from acquire_amazon import BASE, FILES, REVISION
from evaluate_products import DEFAULT_DATA, load_prepared

from memrecagent.amazon import history_before
from memrecagent.amazon_preparation import TEST_END_MS, TRAIN_END_MS, VALIDATION_END_MS
from memrecagent.product_bundle import save_bundle
from memrecagent.product_evaluation import fit_popularity
from memrecagent.product_models import load_fitted_model
from memrecagent.product_service import ProductService

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ROOT / "artifacts/products_models/20261020T045534635541Z-39cd69c1"


def main() -> None:
    """核对源权重/数据版本，只保留截至拟合边界的已见和候选快照。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/product_bundles")
    args = parser.parse_args()
    events, candidates, data = load_prepared(args.data)
    report_path = args.models / "report.json"
    report = json.loads(report_path.read_text())
    if report["data"]["sha256"] != data["sha256"]:
        raise ValueError("模型与处理数据版本不符，未创建推理包")
    data_report = json.loads((args.data / "report.json").read_text())
    cutoffs = report["config"].get("cutoffs_ms")
    if cutoffs is None:
        values = data_report["cutoffs_ms"]
        cutoffs = [values[key] for key in ["train_end", "validation_end", "test_end"]]
    if cutoffs != [TRAIN_END_MS, VALIDATION_END_MS, TEST_END_MS]:
        raise ValueError("模型与数据时间协议版本不符")
    expected = report.get("checkpoint_sha256")
    if expected is None:
        # 首次真实运行的摘要在独立重载审计中，保留旧报告而不重写历史。
        expected = json.loads((args.models / "verification.json").read_text())["checkpoint_sha256"]
    models = {}
    for name in ["mf", "bpr"]:
        path = args.models / f"{name}_test.pt"
        if sha256(path.read_bytes()).hexdigest() != expected[path.name]:
            raise ValueError("源模型SHA256不匹配，未创建推理包")
        models[name] = load_fitted_model(path)
    history = history_before(events, VALIDATION_END_MS)
    counts = fit_popularity(history, candidates)
    if counts != json.loads((args.models / "popularity_test.json").read_text()):
        raise ValueError("热门拟合统计与源模型运行不一致")
    seen: dict[str, set[str]] = {}
    for event in history:
        seen.setdefault(event.user_id, set()).add(event.item_id)
    execution = json.loads((args.data / "execution.json").read_text())
    revision = execution.get("revision")
    if revision == REVISION and data_report.get("input_sha256") != {
        Path(path).name: digest for path, _, digest in FILES
    }:
        raise ValueError("官方来源版本声明与原始哈希不符")
    service = ProductService(
        bundle_id="exporting",
        as_of_ms=VALIDATION_END_MS,
        counts=counts,
        seen=seen,
        products=json.loads((args.data / "products.json").read_text()),
        models=models,
        dataset_revision=revision,
        metadata_url=f"{BASE}/raw/meta_categories/meta_Musical_Instruments.jsonl"
        if revision == REVISION
        else None,
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_root / f"{stamp}-{uuid4().hex[:8]}"
    manifest = save_bundle(
        service,
        output,
        provenance={
            "model_run": str(args.models.resolve()),
            "model_report_sha256": sha256(report_path.read_bytes()).hexdigest(),
            "data_sha256": data["sha256"],
            "selected_epochs": {
                name: row["best_epoch"] for name, row in report["selection"].items()
            },
            "scope": (
                "fixed historical inference; metadata display only; no future labels in bundle"
            ),
        },
    )
    print(
        json.dumps(
            {"bundle": str(output.resolve()), "bundle_id": manifest["bundle_id"]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
