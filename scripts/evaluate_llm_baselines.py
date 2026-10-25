"""同一实际前缀的两个LLM系统基线，默认零调用计划。"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from memrecagent.deepseek_client import DeepSeekClient, ProviderError
from memrecagent.llm_recommendation_baselines import evaluate_baselines, parse_baseline_suite
from memrecagent.memory_extraction import unique_object


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    try:
        raw = args.suite.read_bytes()
        if not 0 < len(raw) <= 8 * 1024 * 1024:
            raise ValueError("size")
        data = json.loads(raw, object_pairs_hook=unique_object)
        cases = parse_baseline_suite(data, split=args.split)
        if args.max_calls is not None and not 2 <= args.max_calls <= 2 * len(cases):
            raise ValueError("budget")
        if not args.live:
            print(
                json.dumps(
                    dict(
                        api_calls=0,
                        planned_cases=len(cases),
                        planned_max_calls=2 * len(cases),
                        mode="plan",
                    )
                )
            )
            return 0
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key or args.max_calls is None or key.encode() in raw or args.output_dir.exists():
            raise ValueError("live configuration")
        client = DeepSeekClient(key, timeout=30)

        def complete(messages):
            response = client.complete(messages)
            if key in json.dumps(asdict(response), ensure_ascii=True):
                raise ProviderError("response_content")
            return response

        report = evaluate_baselines(
            data,
            complete=complete,
            output_dir=args.output_dir,
            max_calls=args.max_calls,
            split=args.split,
            evaluation_kind="synthetic_live",
            input_file=args.suite,
        )
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in (
                        "planned_cases",
                        "client_attempts",
                        "reported_tokens",
                        "unknown_usage_calls",
                        "comparison_complete",
                    )
                }
            )
        )
        return 0 if report["comparison_complete"] else 1
    except (ValueError, TypeError, KeyError, OSError, UnicodeError, RecursionError, ProviderError):
        sys.stderr.write("baseline_unavailable\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
