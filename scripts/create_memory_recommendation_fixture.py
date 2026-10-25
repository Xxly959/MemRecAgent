"""只构造显式合成排名示例；复用固定向量，不训练、不下载。"""

import argparse
import sys
from pathlib import Path

from memrecagent.memory_store import JsonlMemoryStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    path = parser.parse_args().path
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError("合成示例需要空目录，禁止覆盖")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from test_memory import sample_memory
    from test_memory_product_tool import fixture

    fixture(path)
    store = JsonlMemoryStore(path / "memory")
    store.add(sample_memory(entity="品牌A", evidence_text="合成：以后别推荐品牌A"))
    store.add(
        sample_memory(
            memory_id="like-c",
            entity="品牌C",
            relation="prefer",
            evidence_ref="S1:T4",
            evidence_text="合成：我喜欢品牌C",
        )
    )


if __name__ == "__main__":
    main()
