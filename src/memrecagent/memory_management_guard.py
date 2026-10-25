"""统一删除发布期间的共享只读守卫；无存储依赖，避免循环导入。"""

from hashlib import sha256
from pathlib import Path


def management_marker(root: Path, user_id: str) -> Path:
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("用户无效")
    return Path(root) / ".management" / (sha256(user_id.encode()).hexdigest() + ".json")


def ensure_no_pending(root: Path, user_id: str) -> None:
    marker = management_marker(root, user_id)
    if marker.parent.is_symlink() or marker.is_symlink() or marker.exists():
        raise ValueError("management_pending")
