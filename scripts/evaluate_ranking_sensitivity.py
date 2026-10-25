"""完整候选开发集敏感度；默认仅预检，永不调用模型。"""

import argparse
import json
import sys
from pathlib import Path

from memrecagent.ranking_sensitivity import replay_sensitivity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    try:
        result = replay_sensitivity(suite_path=args.suite, output_dir=args.output_dir, run=args.run)
    except (ValueError, OSError, TypeError, RecursionError, KeyError):
        sys.stderr.write("sensitivity_unavailable\n")
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
