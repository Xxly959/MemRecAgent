"""默认只展示三段合成多轮评测计划；--live才请求DeepSeek，最多20次。"""

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from memrecagent.deepseek_client import DeepSeekClient
from memrecagent.memory_dialogue import evaluate_dialogues, load_dialogues

ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    "development": "memory_dialogue_cases.json",
    "acceptance-v2": "memory_dialogue_acceptance_v2.json",
    "acceptance-v3": "memory_dialogue_acceptance_v3.json",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="仅发送冻结合成对话，无自动重试")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--suite", choices=tuple(SUITES), default="development", help="固定合成套件"
    )
    args = parser.parse_args()
    cases = load_dialogues(ROOT / "tests/fixtures" / SUITES[args.suite])
    if not args.live:
        print(
            json.dumps(
                {
                    "mode": "plan",
                    "suite": args.suite,
                    "episodes": 3,
                    "base_steps": 18,
                    "max_model_calls": 20,
                    "api_calls": 0,
                }
            )
        )
        return 0
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("请在本机进程中配置DEEPSEEK_API_KEY，不要发送到聊天或Git。", file=sys.stderr)
        return 2
    output = args.output_dir or ROOT / "artifacts/phase3_dialogue_live" / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
    )
    report = evaluate_dialogues(cases, complete=DeepSeekClient(key).complete, output_dir=output)
    print(json.dumps({"output_dir": str(output), "report": report}, ensure_ascii=False))
    return 0 if report["strict_steps"] == report["base_steps"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
