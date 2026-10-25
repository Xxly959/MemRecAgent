"""共享一次草案的分析对照；默认零调用，live必须显式Key/预算/新输出目录。"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from memrecagent.deepseek_client import DeepSeekClient, ProviderError
from memrecagent.memory_extraction import unique_object
from memrecagent.shared_analysis_evaluation import evaluate_shared_analysis, parse_shared_suite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    try:
        raw = args.suite.read_bytes()
        if not 0 < len(raw) <= 8 * 1024 * 1024:
            raise ValueError("size")
        suite = json.loads(raw, object_pairs_hook=unique_object)
        cases = parse_shared_suite(suite, split=args.split)
        if not 0 <= args.seed < 2**63 or (
            args.max_calls is not None and not 3 <= args.max_calls <= 3 * len(cases)
        ):
            raise ValueError("plan configuration")
        if not args.live:
            print(
                json.dumps(
                    dict(
                        api_calls=0,
                        planned_cases=len(cases),
                        planned_max_calls=3 * len(cases),
                        mode="plan",
                        split=args.split,
                        scope="read-only shared-draft candidate control",
                    )
                )
            )
            return 0
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key or args.max_calls is None or key.encode() in raw or args.output_dir.exists():
            raise ValueError("live configuration")
        client = DeepSeekClient(key)

        def complete(messages):
            response = client.complete(messages)
            if key in json.dumps(asdict(response), ensure_ascii=False):
                raise ProviderError("response_content")
            return response

        report = evaluate_shared_analysis(
            suite,
            complete=complete,
            output_dir=args.output_dir,
            max_calls=args.max_calls,
            split=args.split,
            seed=args.seed,
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
                        "reported_total_tokens",
                        "comparison_ready",
                    )
                }
            )
        )
        return 0 if report["comparison_ready"] else 1
    except (ValueError, TypeError, KeyError, OSError, ProviderError, RecursionError, UnicodeError):
        print(json.dumps(dict(error="invalid_shared_analysis_input")), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
