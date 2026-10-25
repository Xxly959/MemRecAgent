"""本机可信宿主管理的用户用途许可；不由模型或偏好内容推断授权。"""

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from memrecagent.memory import _text
from memrecagent.memory_store import _unique_keys


@dataclass(frozen=True)
class MemoryUsePolicy:
    """个性化读取、自动长期更新和未来广告用途彼此独立。"""

    personalisation: bool = True
    long_term_updates: bool = True
    advertising: bool = False

    def __post_init__(self) -> None:
        if any(type(value) is not bool for value in asdict(self).values()):
            raise ValueError("用途许可必须是明确布尔值")

    def allows_personalisation(self, purpose: str) -> bool:
        """广告必须额外许可，普通推荐许可不自动扩大到广告。"""
        if purpose not in ("recommendation", "advertising"):
            raise ValueError("记忆使用目的无效")
        return self.personalisation and (purpose == "recommendation" or self.advertising)


def _check_purpose(policy: MemoryUsePolicy, purpose: str) -> None:
    if purpose not in ("recommendation", "advertising") or (
        purpose == "advertising" and not policy.allows_personalisation(purpose)
    ):
        raise ValueError("用途未获许可")


class JsonMemoryUseStore:
    """用户hash文件保存设置，revision使旧票据在关开后也不能复活。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, user_id: str) -> Path:
        _text(user_id, "user_id")
        if len(user_id) > 128:
            raise ValueError("用户标识过长")
        return self.root / (sha256(user_id.encode()).hexdigest() + ".json")

    def snapshot(self, user_id: str) -> tuple[MemoryUsePolicy, str]:
        path = self._path(user_id)
        if path.is_symlink():
            raise ValueError("不接受用途设置符号链接")
        try:
            raw = path.read_text()
        except FileNotFoundError:
            return MemoryUsePolicy(), "default"
        try:
            value = json.loads(raw, object_pairs_hook=_unique_keys)
            if (
                not isinstance(value, dict)
                or value.keys() != {"schema_version", "owner_digest", "revision", "policy"}
                or type(value["schema_version"]) is not int
                or value["schema_version"] != 1
                or value["owner_digest"] != path.stem
                or not isinstance(value["revision"], str)
                or len(value["revision"]) != 32
                or any(c not in "0123456789abcdef" for c in value["revision"])
                or not isinstance(value["policy"], dict)
                or value["policy"].keys() != {"personalisation", "long_term_updates", "advertising"}
            ):
                raise ValueError("用途设置无效")
            return MemoryUsePolicy(**value["policy"]), value["revision"]
        except (ValueError, TypeError, KeyError, UnicodeError):
            raise ValueError("用途设置无效") from None

    def get(self, user_id: str) -> MemoryUsePolicy:
        """读取当前用户设置，不创建默认文件。"""
        return self.snapshot(user_id)[0]

    def set(self, user_id: str, policy: MemoryUsePolicy) -> None:
        """同值重设也刷新revision；发布失败保留原文件。"""
        if not isinstance(policy, MemoryUsePolicy):
            raise ValueError("需要明确用途设置")
        self.snapshot(user_id)
        path = self._path(user_id)
        data = dict(
            schema_version=1, owner_digest=path.stem, revision=uuid4().hex, policy=asdict(policy)
        )
        self.root.mkdir(parents=True, exist_ok=True)
        temp = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.root, prefix=".use-policy-", delete=False
            ) as stream:
                temp = Path(stream.name)
                stream.write(json.dumps(data, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, path)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
