"""导出入口必须绑定处理数据与模型版本，包不能依赖未来源文件。"""

import subprocess
import sys
from pathlib import Path

from test_products_cli import prepared_fixture

from memrecagent.product_bundle import load_bundle

ROOT = Path(__file__).resolve().parents[1]


def test_export_binds_versions_and_frozen_bundle_ignores_future_source_changes(tmp_path):
    data, models, bundles = tmp_path / "data", tmp_path / "models", tmp_path / "bundles"
    prepared_fixture(data)
    training = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/train_products.py"),
            "--data",
            str(data),
            "--output-root",
            str(models),
            "--max-epochs",
            "1",
            "--checkpoint-every",
            "1",
            "--factors",
            "2",
            "--batch-size",
            "2",
            "--k",
            "2",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert training.returncode == 0, training.stderr
    model_run = next(models.iterdir())
    command = [
        sys.executable,
        str(ROOT / "scripts/export_products_bundle.py"),
        "--data",
        str(data),
        "--models",
        str(model_run),
        "--output-root",
        str(bundles),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    path = next(bundles.iterdir())
    frozen = load_bundle(path)
    assert "new" not in frozen.seen
    assert frozen.seen["u"] == {"A", "B"}
    before = frozen.recommend("u", bundle_id=frozen.bundle_id, strategy="mf", k=2)
    prepared_fixture(data, test_rating=1)
    after = load_bundle(path).recommend("u", bundle_id=frozen.bundle_id, strategy="mf", k=2)
    assert before == after
    mismatch = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert mismatch.returncode != 0
    assert "版本" in mismatch.stderr
    assert len(list(bundles.iterdir())) == 1
