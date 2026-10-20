"""受控获取必须以真实字节、哈希及文件副作用验收。"""

from hashlib import sha256
from io import BytesIO

import pytest

from memrecagent.downloads import download_verified


def test_valid_stream_is_published_only_after_size_and_digest_match(tmp_path):
    target = tmp_path / "data.jsonl"
    result = download_verified(
        "https://example.invalid/data",
        target,
        3,
        sha256(b"abc").hexdigest(),
        max_bytes=3,
        open_url=lambda *args, **kwargs: BytesIO(b"abc"),
    )
    assert target.read_bytes() == b"abc"
    assert result["bytes"] == 3
    assert result["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert result["cached"] is False
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize("body", [b"ab", b"abcd", b"xyz"])
def test_truncated_oversized_or_wrong_digest_stream_never_becomes_final_file(tmp_path, body):
    target = tmp_path / "data.jsonl"
    with pytest.raises(ValueError):
        download_verified(
            "https://example.invalid/data",
            target,
            3,
            sha256(b"abc").hexdigest(),
            max_bytes=3,
            open_url=lambda *args, **kwargs: BytesIO(body),
        )
    assert not target.exists()
    assert len(list(tmp_path.glob("*.part"))) == 1


def test_declared_budget_failure_happens_before_network_or_file_creation(tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("超预算不应请求网络")

    with pytest.raises(ValueError):
        download_verified(
            "https://example.invalid/data",
            tmp_path / "file",
            100,
            "a" * 64,
            max_bytes=99,
            open_url=forbidden,
        )
    assert list(tmp_path.iterdir()) == []


def test_matching_cache_is_reused_without_network_and_mismatched_cache_is_preserved(tmp_path):
    target = tmp_path / "data"
    target.write_bytes(b"abc")

    def forbidden(*args, **kwargs):
        raise AssertionError("缓存不应重新下载")

    assert (
        download_verified(
            "https://example.invalid/data",
            target,
            3,
            sha256(b"abc").hexdigest(),
            max_bytes=3,
            open_url=forbidden,
        )["cached"]
        is True
    )
    with pytest.raises(ValueError):
        download_verified(
            "https://example.invalid/data",
            target,
            3,
            sha256(b"xyz").hexdigest(),
            max_bytes=3,
            open_url=forbidden,
        )
    assert target.read_bytes() == b"abc"


def test_concurrent_final_file_is_never_overwritten(tmp_path):
    target = tmp_path / "data"

    def opening(*args, **kwargs):
        target.write_bytes(b"other")
        return BytesIO(b"abc")

    with pytest.raises(FileExistsError):
        download_verified(
            "https://example.invalid/data",
            target,
            3,
            sha256(b"abc").hexdigest(),
            max_bytes=3,
            open_url=opening,
        )
    assert target.read_bytes() == b"other"
