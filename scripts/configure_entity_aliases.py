"""可信操作者配置实体表；默认只预检，只能显式初始化新空命名空间。"""

import argparse
import json
import sys
from pathlib import Path

from memrecagent.entity_aliases import (
    MAX_REGISTRY_BYTES,
    decode_entity_aliases,
    install_entity_aliases,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--initialize", action="store_true")
    args = parser.parse_args()
    try:
        with args.input.open("rb") as stream:
            table = decode_entity_aliases(stream.read(MAX_REGISTRY_BYTES + 1))
        if args.initialize:
            install_entity_aliases(args.root, table.to_document())
        print(
            json.dumps(
                dict(
                    mode="initialized" if args.initialize else "plan",
                    registry_digest=table.digest,
                    n_entities=len(table.entries),
                    api_calls=0,
                ),
                ensure_ascii=False,
            )
        )
        return 0
    except (ValueError, TypeError, OSError):
        print(
            json.dumps(dict(error="invalid_or_existing_entity_namespace", api_calls=0)),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
