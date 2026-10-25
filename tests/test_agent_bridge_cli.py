"""有界二进制帧及真实子进程，不以关闭stdin掩盖flush问题。"""

import io
import json
import selectors
import subprocess
import sys

import pytest
from test_agent_bridge import error, request
from test_product_service import service

from memrecagent.agent_bridge import CatalogBridge, serve_bridge
from memrecagent.product_bundle import save_bundle


def frame(value):
    return json.dumps(value, ensure_ascii=False).encode() + b"\n"


def run_bytes(raw):
    output = io.BytesIO()
    code = serve_bridge(CatalogBridge(service()), io.BytesIO(raw), output)
    return code, [json.loads(line) for line in output.getvalue().splitlines()]


@pytest.mark.parametrize(
    "raw",
    [
        b"\n",
        b"not json\n",
        b"\xff\n",
        b'{"version":1,"id":"r1","method":"catalog_status","params":{},"id":"r2"}\n',
        b'{"version":1,"id":"r1","method":"catalog_status","params":{"a":1,"a":2}}\n',
        b'{"version":NaN}\n',
        b'{"version":Infinity}\n',
        b'{"version":-Infinity}\n',
        b"[" * 20000 + b"0" + b"]" * 20000 + b"\n",
    ],
    ids=[
        "empty",
        "syntax",
        "utf8",
        "duplicate",
        "nested-duplicate",
        "nan",
        "infinity",
        "negative-infinity",
        "decoder-recursion",
    ],
)
def test_invalid_json_is_sanitized_and_next_frame_still_runs(raw):
    code, replies = run_bytes(raw + frame(request()))
    assert code == 0
    assert replies[0] == error("invalid_json", None)
    assert replies[1]["id"] == "r1" and replies[1]["result"]["n_catalog_items"] == 3


@pytest.mark.parametrize("ending", [b"\n", b"\r\n"])
@pytest.mark.parametrize("prefix", [b"", " 中文".encode()])
def test_exact_byte_limit_accepts_ascii_and_multibyte_payloads(ending, prefix):
    # 中文放在未知方法中，仍应完成JSON解析，而非被错误按字符判定帧长。
    value = request(method="catalog_status" if not prefix else "中文")
    payload = frame(value)[:-1]
    payload += b" " * (65536 - len(payload))
    code, replies = run_bytes(payload + ending)
    assert code == 0
    if prefix:
        assert replies == [error("unknown_method")]
    else:
        assert replies[0]["ok"] is True


@pytest.mark.parametrize("ending", [b"\n", b"\r\n", b""])
def test_oversize_frame_terminates_without_executing_appended_tail(ending):
    payload = frame(request())[:-1]
    payload += b" " * (65537 - len(payload))
    code, replies = run_bytes(payload + ending + frame(request(id="tail")))
    assert code == 2
    assert replies == [error("frame_too_large", None)]


def test_empty_eof_is_clean_but_unterminated_frame_is_fatal():
    assert run_bytes(b"") == (0, [])
    assert run_bytes(frame(request())[:-1]) == (2, [error("frame_truncated", None)])
    assert run_bytes(b" " * 65536 + b"\r") == (2, [error("frame_truncated", None)])


def test_bounded_read_and_flush_after_each_response():
    class BoundedInput(io.BytesIO):
        def readline(self, size=-1):
            assert 0 < size <= 65539
            return super().readline(size)

    class FlushedOutput(io.BytesIO):
        def __init__(self):
            super().__init__()
            self.frames = []

        def flush(self):
            self.frames.append(self.getvalue())

    output = FlushedOutput()
    assert (
        serve_bridge(
            CatalogBridge(service()),
            BoundedInput(frame(request()) + frame(request(id="r2"))),
            output,
        )
        == 0
    )
    assert [len(item.splitlines()) for item in output.frames] == [1, 2]


def read_reply(process):
    # 用select避免子进程忘记flush时测试本身永久阻塞。
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        assert selector.select(timeout=10), "子进程未及时交付回执"
    line = process.stdout.readline()
    assert line, "子进程提前关闭stdout"
    return json.loads(line)


@pytest.mark.parametrize("size", [65537, 65538])
def test_known_oversize_replies_without_newline_or_closed_stdin(tmp_path, size):
    path = tmp_path / "bundle"
    save_bundle(service(), path, provenance={"kind": "synthetic"})
    process = subprocess.Popen(
        [sys.executable, "-m", "memrecagent.agent_bridge", "--bundle", str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        process.stdin.write(b" " * size)
        process.stdin.flush()
        # 已经确定超限，不能要求发送方再补LF或关闭输入才给错误。
        assert read_reply(process) == error("frame_too_large", None)
        assert process.wait(timeout=10) == 2
        assert process.stderr.read() == b""
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()


def test_real_process_loads_once_and_replies_before_stdin_closes(tmp_path):
    bound = tmp_path / "bound"
    other = tmp_path / "other"
    manifest = save_bundle(service(), bound, provenance={"kind": "synthetic"})
    save_bundle(service(), other, provenance={"kind": "synthetic"})
    process = subprocess.Popen(
        [sys.executable, "-m", "memrecagent.agent_bridge", "--bundle", str(bound)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    try:
        process.stdin.write(frame(request()))
        process.stdin.flush()
        first = read_reply(process)
        assert first["result"]["bundle_id"] == manifest["bundle_id"]
        assert first["result"]["n_catalog_items"] == 3
        # 首次加载后，磁盘变化和模型给出的路径都不能替换绑定实例。
        (other / "service.json").write_text("corrupt other")
        (bound / "service.json").write_text("corrupt bound after load")
        process.stdin.write(frame(request(id="switch", params={"path": str(other)})))
        process.stdin.flush()
        assert read_reply(process) == error("invalid_request", "switch")
        process.stdin.write(frame(request(id="r2")))
        process.stdin.flush()
        second = read_reply(process)
        assert second["result"] == first["result"]
        process.stdin.close()
        assert process.wait(timeout=10) == 0
        assert process.stderr.read() == b""
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()


@pytest.mark.parametrize("corrupt", [False, True])
def test_startup_bundle_failure_is_fixed_and_stdout_empty(tmp_path, corrupt):
    path = tmp_path / "sensitive-looking-path"
    if corrupt:
        save_bundle(service(), path, provenance={"kind": "synthetic"})
        (path / "mf.pt").write_bytes(b"corrupt")
    result = subprocess.run(
        [sys.executable, "-m", "memrecagent.agent_bridge", "--bundle", str(path)],
        input=frame(request()),
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == b"bundle_load_failed\n"
