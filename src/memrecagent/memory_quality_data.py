"""作者合成语料契约；检查标签内在一致性，不将标签传给分析模型。"""

import json
import math
import re
import unicodedata
from pathlib import Path

from memrecagent.memory_extraction import unique_object

RELATIONS = {"prefer", "dislike", "exclude", "neutral"}
ACTIONS = {"continue", "answer", "compare", "recommend", "clarify", "feedback_target_required"}
EXPECTED_FIELDS = {
    "writes",
    "heads",
    "activated",
    "actions",
    "other_context",
    "other_only",
    "current_only",
    "quoted_only",
    "revisions",
}


def _text(value: object, limit: int = 4000) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= limit


def _facts(value: object, width: int = 3) -> set[tuple]:
    if not isinstance(value, list) or len(value) > 32:
        raise ValueError("状态参考需要有限列表")
    result = set()
    for row in value:
        if (
            not isinstance(row, list)
            or len(row) != width
            or any(not _text(cell) for cell in row)
            or row[-3] != "phone"
            or row[-1] not in RELATIONS
        ):
            raise ValueError("状态参考字段无效")
        result.add(tuple(row))
    if len(result) != len(value):
        raise ValueError("重复状态不能增加样本数")
    return result


def validate_quality_suite(value: dict) -> None:
    """从空库核对gold修订/删除/时效，拒绝模板跨划分及未发生的确认。"""
    if (
        not isinstance(value, dict)
        or value.keys() != {"schema_version", "synthetic", "label_source", "episodes"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["synthetic"] is not True
        or not _text(value["label_source"])
        or not isinstance(value["episodes"], list)
        or not 1 <= len(value["episodes"]) <= 200
    ):
        raise ValueError("只接受版本1显式合成语料")
    episode_ids, family_splits, story_splits = set(), {}, {}
    for episode in value["episodes"]:
        if not isinstance(episode, dict) or episode.keys() != {
            "episode_id",
            "template_family",
            "split",
            "scenario",
            "people",
            "max_memory_age_seconds",
            "steps",
        }:
            raise ValueError("故事字段无效")
        for key in ("episode_id", "template_family", "scenario"):
            if not isinstance(episode[key], str) or not re.fullmatch(
                r"[A-Za-z0-9_-]{1,64}", episode[key]
            ):
                raise ValueError("故事标识无效")
        identity, split = episode["episode_id"], episode["split"]
        if identity in episode_ids or split not in ("dev", "test"):
            raise ValueError("故事重复或划分无效")
        episode_ids.add(identity)
        if family_splits.setdefault(episode["template_family"], split) != split:
            raise ValueError("同模板族不能跨开发与测试")
        age = episode["max_memory_age_seconds"]
        if age is not None and (
            type(age) not in (int, float) or not math.isfinite(age) or age <= 0
        ):
            raise ValueError("记忆年龄无效")
        people = episode["people"]
        if not isinstance(people, list) or len(people) > 16:
            raise ValueError("人物参考无效")
        aliases, person_ids = set(), set()
        for person in people:
            if (
                not isinstance(person, dict)
                or person.keys() != {"person_id", "aliases"}
                or not _text(person["person_id"], 64)
                or person["person_id"] in person_ids
                or not isinstance(person["aliases"], list)
                or not person["aliases"]
                or any(not _text(a, 128) or a in aliases for a in person["aliases"])
                or len(set(person["aliases"])) != len(person["aliases"])
            ):
                raise ValueError("人物别名必须明确且不重叠")
            aliases.update(person["aliases"])
            person_ids.add(person["person_id"])
        steps = episode["steps"]
        if not isinstance(steps, list) or not 1 <= len(steps) <= 32:
            raise ValueError("每故事需要1至32步")
        previous_time, last_session = -10, None
        sessions, turns, gold = set(), {}, {}
        for step in steps:
            if not isinstance(step, dict) or step.keys() != {
                "turn_id",
                "session_id",
                "offset_seconds",
                "text",
                "reply_to",
                "trusted_delete",
                "assistant",
                "expected",
                "tags",
            }:
                raise ValueError("轮次字段无效")
            if (
                not isinstance(step["tags"], list)
                or not step["tags"]
                or len(step["tags"]) > 12
                or any(not _text(tag, 64) for tag in step["tags"])
                or len(set(step["tags"])) != len(step["tags"])
            ):
                raise ValueError("语义标签无效")
            turn_id, session = step["turn_id"], step["session_id"]
            if (
                not _text(turn_id, 64)
                or turn_id in turns
                or not _text(session, 64)
                or type(step["offset_seconds"]) is not int
                or not previous_time + 5 < step["offset_seconds"] <= 31536000
                or not _text(step["text"])
            ):
                raise ValueError("轮次标识、文本或时间无效")
            if session != last_session:
                if session in sessions:
                    raise ValueError("关闭的会话不能恢复旧RAM")
                sessions.add(session)
            reply = step["reply_to"]
            if reply is not None and (
                reply not in turns
                or turns[reply] != session
                or reply != next(reversed(turns), None)
            ):
                raise ValueError("条件回复必须紧接同会话实际问题")
            turns[turn_id] = session
            previous_time, last_session = step["offset_seconds"], session
            deleted = step["trusted_delete"]
            if deleted is not None:
                if not _text(deleted):
                    raise ValueError("可信删除实体无效")
                gold.pop(("phone", deleted), None)
            assistant = step["assistant"]
            if assistant is not None and (
                not isinstance(assistant, dict)
                or assistant.keys() != {"on_action", "text"}
                or assistant["on_action"] not in ACTIONS - {"clarify"}
                or not _text(assistant["text"])
            ):
                raise ValueError("脚本助手字段无效")
            expected = step["expected"]
            if not isinstance(expected, dict) or expected.keys() != EXPECTED_FIELDS:
                raise ValueError("参考字段无效")
            if (
                not isinstance(expected["actions"], list)
                or not expected["actions"]
                or any(a not in ACTIONS for a in expected["actions"])
                or len(set(expected["actions"])) != len(expected["actions"])
            ):
                raise ValueError("参考动作无效")
            facts = {
                key: _facts(expected[key]) for key in EXPECTED_FIELDS - {"actions", "other_context"}
            }
            other = _facts(expected["other_context"], 4)
            if any(r[0] not in person_ids for r in other):
                raise ValueError("他人上下文缺少人物参考")
            if facts["writes"] & (
                facts["other_only"] | facts["current_only"] | facts["quoted_only"]
            ):
                raise ValueError("同事实不能既应长期写入又禁止长期写入")
            if not facts["revisions"] <= facts["writes"]:
                raise ValueError("修订必须是本轮写入")
            for row in facts["revisions"]:
                if row[:2] not in gold:
                    raise ValueError("修订不能引用未来或不存在的偏好")
            if len({r[:2] for r in facts["writes"]}) != len(facts["writes"]):
                raise ValueError("同轮不能写入冲突态度")
            for row in facts["writes"]:
                gold[row[:2]] = (row, step["offset_seconds"])
            visible = {
                row
                for row, occurred in gold.values()
                if age is None or step["offset_seconds"] + 1 - occurred < age
            }
            if facts["heads"] != visible:
                raise ValueError("参考状态与历史写入/修订/删除/时效不一致")
        story = unicodedata.normalize("NFKC", "\n".join(s["text"] for s in steps)).casefold()
        if story_splits.setdefault(story, split) != split:
            raise ValueError("相同故事不能跨开发与测试")


def load_quality_suite(path: Path, *, split: str = "all") -> dict:
    """先校验完整语料再筛选，不因筛选掩盖跨划分泄漏。"""
    raw = path.read_bytes()
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("语料过大")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
    validate_quality_suite(value)
    if split not in ("all", "dev", "test"):
        raise ValueError("未知划分")
    if split != "all":
        value["episodes"] = [e for e in value["episodes"] if e["split"] == split]
    validate_quality_suite(value)
    return value
