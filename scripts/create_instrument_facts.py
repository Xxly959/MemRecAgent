"""从既有乐器包生成有来源的精确商品ID映射；不训练、不下载、不覆盖。"""

import argparse
import json
from pathlib import Path

from memrecagent.instrument_catalog import build_instrument_mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mapping = build_instrument_mapping(args.bundle)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(mapping, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")


if __name__ == "__main__":
    main()
