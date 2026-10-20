"""按大小与SHA256验证获取文件；不覆盖缓存，不发布未验证内容。"""

import logging
import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlsplit
from urllib.request import urlopen
from uuid import uuid4

logger = logging.getLogger(__name__)
CHUNK_BYTES = 1024 * 1024


def download_verified(
    url: str,
    destination: Path,
    expected_size: int,
    expected_sha256: str,
    *,
    max_bytes: int,
    open_url: Callable[..., AbstractContextManager[BinaryIO]] = urlopen,
) -> dict:
    """先验证预算，再流式校验；失败保留.part供审查，不自动重试计费。"""
    if (
        type(expected_size) is not int
        or type(max_bytes) is not int
        or not 0 < expected_size <= max_bytes
        or len(expected_sha256) != 64
        or any(c not in "0123456789abcdef" for c in expected_sha256)
        or urlsplit(url).scheme != "https"
    ):
        raise ValueError("获取需要HTTPS、合法SHA256、正大小与足够预算")
    digest = sha256()
    if destination.exists():
        if destination.stat().st_size != expected_size:
            raise ValueError("已有缓存大小不匹配，保留原文件")
        with destination.open("rb") as stream:
            while chunk := stream.read(CHUNK_BYTES):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("已有缓存SHA256不匹配，保留原文件")
        return {"bytes": expected_size, "sha256": digest.hexdigest(), "cached": True}

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.{uuid4().hex}.part")
    size = 0
    progress_bucket = 0
    with partial.open("xb") as output, open_url(url, timeout=40) as source:
        while chunk := source.read(min(CHUNK_BYTES, expected_size - size + 1)):
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
            if size > expected_size:
                raise ValueError("响应超过已核验大小，未发布目标文件")
            bucket = size // (64 * CHUNK_BYTES)
            if bucket > progress_bucket:
                logger.info("获取 %s: %d/%d bytes", destination.name, size, expected_size)
                progress_bucket = bucket
        output.flush()
        os.fsync(output.fileno())
    if size != expected_size or digest.hexdigest() != expected_sha256:
        raise ValueError("响应大小或SHA256不匹配，保留.part而不发布目标文件")
    # link为原子不覆盖发布：即使并发产生同名目标，也不能替换对方文件。
    os.link(partial, destination)
    partial.unlink()
    return {"bytes": size, "sha256": digest.hexdigest(), "cached": False}
