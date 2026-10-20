"""处理固定官方文件，保存去正文事件、展示目录与训练期候选。"""

import argparse
import json
import platform
import resource
import sys
import time
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from acquire_amazon import FILES, REVISION

from memrecagent.amazon_preparation import prepare_amazon


def main() -> None:
    """仅处理已完整校验数据，使用新目录并保存代码/环境/资源证据。"""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=root / "data/raw/amazon-2023" / REVISION)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.raw / "manifest.json").read_text())
    if manifest.get("revision") != REVISION or not manifest.get("all_files_verified"):
        raise ValueError("需要固定版本完整获取清单；不自动下载或使用.part")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or root / "data/processed/amazon-2023" / f"{stamp}-{uuid4().hex[:8]}"
    expected = {Path(path).name: digest for path, _, digest in FILES}
    started = time.perf_counter()
    report = prepare_amazon(
        args.raw / "Musical_Instruments.jsonl",
        args.raw / "meta_Musical_Instruments.jsonl",
        output,
        expected_hashes=expected,
    )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record = {
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
        "environment": {"python": sys.version, "platform": platform.platform()},
        "revision": REVISION,
        "source_sha256": {
            str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__),
                root / "scripts/acquire_amazon.py",
                root / "src/memrecagent/amazon.py",
                root / "src/memrecagent/amazon_preparation.py",
            ]
        },
        "output_sha256": {
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.iterdir())
            if path.is_file()
        },
    }
    with (output / "execution.json").open("x") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {"output": str(output), "report": report, "execution": record},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
