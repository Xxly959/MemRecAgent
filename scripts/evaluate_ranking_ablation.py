"""默认展示计划；--run-offline冻结并执行四组排名，API调用始终为0。"""

import argparse
import json
import sys
from pathlib import Path

from memrecagent.memory_ablation_replay import replay_ablation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--memory-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--normalisation", choices=("minmax", "rank"), default="minmax")
    parser.add_argument("--run-offline", action="store_true")
    args = parser.parse_args()
    try:
        report = replay_ablation(
            bundle=args.bundle,
            facts_path=args.facts,
            suite_path=args.suite,
            memory_root=args.memory_root,
            output_dir=args.output_dir,
            normalisation=args.normalisation,
            run=args.run_offline,
        )
    except (ValueError, TypeError, KeyError, OSError, OverflowError):
        print(json.dumps({"error": "invalid_ablation_input"}), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False))
    return int(bool(report.get("n_execution_errors")) or report.get("inputs_unchanged") is False)


if __name__ == "__main__":
    raise SystemExit(main())
