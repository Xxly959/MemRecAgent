"""获取已审计的固定类别；默认只报告计划，显式--fetch才访问数据。"""

import argparse
import json
import logging
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from memrecagent.downloads import download_verified

REVISION = "2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e"
BASE = f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/{REVISION}"
FILES = [
    (
        "raw/review_categories/Musical_Instruments.jsonl",
        1557845107,
        "63aee24d5b9eaf0069cf638703e2d62eb80fde5d2c00ca3d29881d645f6214b3",
    ),
    (
        "raw/meta_categories/meta_Musical_Instruments.jsonl",
        631877970,
        "96f40b5440918f07e3e08e84b7a6874a06a5e32a9988998ca9520cfd1b51a12c",
    ),
]


def main() -> None:
    """保存来源和用途；完成完整字节校验后才写获取完成清单。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--max-bytes", type=int, default=2_500_000_000)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/raw/amazon-2023" / REVISION,
    )
    args = parser.parse_args()
    total = sum(size for _, size, _ in FILES)
    if args.max_bytes < total:
        parser.error("总获取预算不足；未访问数据或创建目录")
    files = {
        Path(path).name: {"url": f"{BASE}/{path}", "bytes": size, "sha256": digest}
        for path, size, digest in FILES
    }
    plan = {"revision": REVISION, "total_bytes": total, "files": files, "fetch": args.fetch}
    if not args.fetch:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.destination.mkdir(parents=True, exist_ok=True)
    intent = {
        **plan,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "仅本地学习研究；不再分发原始数据、评论或图片",
        "terms_source": (
            "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/discussions/1"
        ),
        "license": "发布方无权授予数据许可证；不得套用代码MIT许可",
    }
    intent_path = args.destination / "acquisition-intent.json"
    if not intent_path.exists():
        with intent_path.open("x") as stream:
            json.dump(intent, stream, ensure_ascii=False, indent=2)
    else:
        previous = json.loads(intent_path.read_text())
        if previous.get("revision") != REVISION or previous.get("files") != files:
            raise ValueError("已有获取意图与固定来源不符，未覆盖")
    for filename, spec in files.items():
        result = download_verified(
            spec["url"],
            args.destination / filename,
            spec["bytes"],
            spec["sha256"],
            max_bytes=args.max_bytes,
        )
        logging.info("已校验 %s: bytes=%d cached=%s", filename, result["bytes"], result["cached"])
    manifest = {
        **intent,
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "all_files_verified": True,
    }
    manifest_path = args.destination / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("files") != files or previous.get("revision") != REVISION:
            raise ValueError("已有来源清单不匹配，未覆盖")
    else:
        with manifest_path.open("x") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"manifest": str(manifest_path), "verified_bytes": total}, ensure_ascii=False))


if __name__ == "__main__":
    main()
