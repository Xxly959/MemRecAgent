"""默认仅展示合成评测计划；--live才使用本机环境中的DeepSeek凭证。"""

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from memrecagent.deepseek_client import DeepSeekClient
from memrecagent.memory_extraction_evaluation import evaluate_cases, load_cases

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="真实调用最多24次，仅发送合成案例")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    cases = load_cases(ROOT / "tests/fixtures/memory_extraction_cases.json")
    if not args.live:
        print(
            json.dumps(
                {
                    "mode": "plan",
                    "cases": len(cases),
                    "api_calls": 0,
                    "model": "deepseek-flash",
                    "thinking": "disabled",
                }
            )
        )
        return 0
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("请在本机运行进程中配置DEEPSEEK_API_KEY；不要发送到聊天或Git。", file=sys.stderr)
        return 2
    client = DeepSeekClient(key)
    output = args.output_dir or ROOT / "artifacts/phase3_language_live" / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
    )
    report = evaluate_cases(cases, complete=client.complete, output_dir=output)
    print(json.dumps({"output_dir": str(output), "report": report}, ensure_ascii=False))
    return 0 if report["strict_correct"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
