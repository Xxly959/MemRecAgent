"""版本化语言评测；默认只校验计划，live需要环境Key和显式调用上限。"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from memrecagent.deepseek_client import DeepSeekClient
from memrecagent.memory_quality_data import load_quality_suite
from memrecagent.memory_quality_evaluation import evaluate_memory_quality

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", type=Path, default=ROOT / "tests/fixtures/memory_quality_v1.json"
    )
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument(
        "--analysis-mode", choices=("direct", "analyst", "analyst_blind"), default="direct"
    )
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    def fail(code):
        print(json.dumps({"error": code}), file=sys.stderr)
        return 2

    try:
        suite = load_quality_suite(args.suite, split=args.split)
        steps = [s for e in suite["episodes"] for s in e["steps"]]
        upper = len(steps) * (1 if args.analysis_mode == "direct" else 2)
        if not args.live:
            print(
                json.dumps(
                    dict(
                        mode="plan",
                        split=args.split,
                        analysis_mode=args.analysis_mode,
                        n_episodes=len(suite["episodes"]),
                        n_steps=len(steps),
                        max_provider_attempts=upper,
                        api_calls=0,
                        tag_counts=dict(Counter(t for s in steps for t in s["tags"])),
                        label_source=suite["label_source"],
                    ),
                    ensure_ascii=False,
                )
            )
            return 0
        if args.max_calls is None:
            return fail("explicit_budget_required")
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            return fail("missing_environment_key")
        report = evaluate_memory_quality(
            suite,
            complete=DeepSeekClient(key, timeout=20).complete,
            output_dir=args.output_dir,
            analysis_mode=args.analysis_mode,
            max_calls=args.max_calls,
            evaluation_kind="synthetic_live",
        )
    except (ValueError, TypeError, KeyError, OSError, OverflowError):
        return fail("invalid_quality_input")
    print(json.dumps(dict(output_dir=str(args.output_dir), report=report), ensure_ascii=False))
    if (
        not report["inputs_unchanged"]
        or report["n_provider_errors"]
        or report["n_preflight_errors"]
    ):
        return 2
    return 0 if report["strict_steps"] == report["n_steps"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
