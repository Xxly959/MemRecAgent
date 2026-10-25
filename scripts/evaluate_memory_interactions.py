"""默认只展示验收计划；显式--live调用DeepSeek，direct20/analyst40次，无重试。"""

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from memrecagent.deepseek_client import DeepSeekClient
from memrecagent.memory_interaction_evaluation import evaluate_interactions, load_interactions

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="仅发送冻结合成对话")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--analysis-mode", choices=("direct", "analyst"), default="direct")
    parser.add_argument(
        "--suite", choices=("progressive-v1", "analyst-v1"), default="progressive-v1"
    )
    args = parser.parse_args()
    filename = (
        "memory_analyst_cases_v1.json"
        if args.suite == "analyst-v1"
        else "memory_interaction_cases_v1.json"
    )
    cases = load_interactions(ROOT / "tests/fixtures" / filename)
    if not args.live:
        print(
            json.dumps(
                dict(
                    mode="plan",
                    suite=args.suite,
                    analysis_mode=args.analysis_mode,
                    episodes=3,
                    base_steps=18,
                    max_model_calls=40 if args.analysis_mode == "analyst" else 20,
                    api_calls=0,
                    assistant_source="scripted_assistant_context",
                )
            )
        )
        return 0
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("请通过本机环境配置DEEPSEEK_API_KEY，不要发送到聊天。", file=sys.stderr)
        return 2
    output = args.output_dir or ROOT / "artifacts/phase3_interactions_live" / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
    )
    report = evaluate_interactions(
        cases,
        complete=DeepSeekClient(key).complete,
        output_dir=output,
        analysis_mode=args.analysis_mode,
    )
    print(json.dumps(dict(output_dir=str(output), report=report), ensure_ascii=False))
    return 0 if report["strict_steps"] == 18 else 1


if __name__ == "__main__":
    raise SystemExit(main())
