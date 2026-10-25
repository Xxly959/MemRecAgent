"""本地本人记忆管理；显式查看原文、纠正或删除，不调用模型。"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from memrecagent.memory_management import MemoryManager
from memrecagent.memory_store import JsonlMemoryStore, _unique_keys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="本机私有记忆目录")
    parser.add_argument("--user-id", required=True, help="本地数据归属；不是身份认证")
    commands = parser.add_subparsers(dest="command", required=True)
    view = commands.add_parser("list", help="显式查看证据原文，请在私有终端操作")
    view.add_argument("--as-of", required=True, help="带时区的查询时刻，严格早于边界")
    view.add_argument("--category")
    view.add_argument("--entity")
    view.add_argument("--history", action="store_true", help="查看完整历史而非当前版本")
    correction = commands.add_parser("correct", help="根据用户明确提供的新证据纠正")
    correction.add_argument("--input", type=Path, required=True, help="私有JSON输入文件")
    deletion = commands.add_parser("delete", help="删除指定类别/对象的全部偏好历史")
    deletion.add_argument("--category", required=True)
    deletion.add_argument("--entity", required=True)
    deletion.add_argument("--at", required=True, help="带时区的删除时刻")
    inspect = commands.add_parser("inspect", help="显式私有查看品牌/属性完整历史与当前反馈")
    inspect.add_argument("--as-of", required=True)
    for name in ("forget", "forget-feedback"):
        forget = commands.add_parser(name, help="统一删除精确范围并保存防重放水位；后者仅删反馈")
        forget.add_argument(
            "--kind",
            choices=("brand", "product")
            if name == "forget-feedback"
            else ("brand", "product", "attribute"),
            required=True,
        )
        forget.add_argument("--category", required=True)
        forget.add_argument("--target", required=True)
        forget.add_argument("--at", required=True)
    commands.add_parser("recover", help="完成此前中断的删除发布，不调用模型")
    args = parser.parse_args()
    store = JsonlMemoryStore(args.root)
    try:
        if args.command == "recover":
            result = {"recovered": MemoryManager(args.root).recover(args.user_id)}
        elif args.command == "inspect":
            result = MemoryManager(args.root).inspect(
                args.user_id, as_of=datetime.fromisoformat(args.as_of)
            )
        elif args.command in {"forget", "forget-feedback"}:
            result = MemoryManager(args.root).forget(
                args.user_id,
                kind=args.kind,
                category=args.category,
                target=args.target,
                at=datetime.fromisoformat(args.at),
                feedback_only=args.command == "forget-feedback",
            )
        elif args.command == "list":
            read = store.list_memories if args.history else store.active_memories
            result = [
                m.to_dict()
                for m in read(
                    args.user_id,
                    category=args.category,
                    entity=args.entity,
                    as_of=datetime.fromisoformat(args.as_of),
                )
            ]
        elif args.command == "correct":
            data = json.loads(args.input.read_text(), object_pairs_hook=_unique_keys)
            expected = {
                "category",
                "entity",
                "expected_head",
                "relation",
                "evidence_ref",
                "evidence_text",
                "occurred_at",
                "available_at",
            }
            if not isinstance(data, dict) or data.keys() != expected:
                raise ValueError("纠正字段不完整")
            for name in ("occurred_at", "available_at"):
                if not isinstance(data[name], str):
                    raise ValueError("时间类型无效")
                data[name] = datetime.fromisoformat(data[name])
            memory = store.correct(args.user_id, **data)
            result = {"status": "corrected", "memory_id": memory.memory_id}
        else:
            count = store.delete_entity(
                args.user_id,
                category=args.category,
                entity=args.entity,
                deleted_at=datetime.fromisoformat(args.at),
            )
            result = {"deleted_records": count}
    except (ValueError, OSError):
        # 错误输出不回显原话、输入文件内容和内部堆栈。
        print(json.dumps({"error": "invalid_input_or_store"}), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
