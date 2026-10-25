"""仅供离线宿主测试构造既有固定向量包，不训练或下载。"""

import argparse
import sys
from pathlib import Path

from memrecagent.product_bundle import save_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from test_product_service import service

    save_bundle(service(), args.path, provenance={"kind": "synthetic-pi-test"})


if __name__ == "__main__":
    main()
