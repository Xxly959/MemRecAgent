"""导出已核验目录的唯一完整标题别名及覆盖报告；不自动安装或覆盖。"""

import argparse
import json
from pathlib import Path

from memrecagent.instrument_catalog import build_instrument_title_aliases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("保留已有标题别名导出目录")
    document, report = build_instrument_title_aliases(args.bundle)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in (("entity-aliases.json", document), ("report.json", report)):
        with (args.output_dir / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")


if __name__ == "__main__":
    main()
