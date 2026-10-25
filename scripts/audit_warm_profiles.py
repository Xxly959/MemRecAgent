"""默认零调用计划；已有真实包的结构化反事实机制审计，不训练。"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from memrecagent.warm_profile_audit import run_profile_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--active-count", type=int, default=16)
    parser.add_argument("--cold-count", type=int, default=4)
    parser.add_argument("--run-offline", action="store_true")
    args = parser.parse_args()
    result = run_profile_audit(
        bundle=args.bundle,
        output=args.output_dir,
        count=args.active_count,
        cold_count=args.cold_count,
        as_of=datetime.fromisoformat(args.as_of),
        category=args.category,
        run=args.run_offline,
    )
    print(json.dumps(result, ensure_ascii=False))
    return int(
        bool(result.get("n_execution_errors"))
        or result.get("inputs_and_source_unchanged") is False
        or result.get("loaded_parameters_unchanged") is False
    )


if __name__ == "__main__":
    raise SystemExit(main())
