"""启动自有本地子进程核验真实HTTP，结束后关闭，不留下后台服务。"""

import argparse
import json
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import torch
from evaluate_movielens import save_json

from memrecagent.product_bundle import load_bundle

ROOT = Path(__file__).resolve().parents[1]


def request_json(url: str, payload: dict | None = None) -> tuple[int, dict]:
    """仅请求调用方给定的本机地址，不记录请求中的用户标识。"""
    request = Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        return exc.code, json.load(exc)


def main() -> None:
    """比较真实HTTP与Python结果，只报告烟测，不外推吞吐或延迟分位数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/product_api_http")
    args = parser.parse_args()
    torch.set_num_threads(1)
    service = load_bundle(args.bundle)
    created = datetime.now(UTC)
    run = args.output_root / f"{created.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:8]}"
    run.mkdir(parents=True, exist_ok=False)
    process = None
    report = {
        "created_at_utc": created.isoformat(),
        "bundle_id": service.bundle_id,
        "status": "FAIL",
        "host": "127.0.0.1",
        "cases": {},
        "scope": "local HTTP smoke; not latency/throughput benchmark",
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    responses = {}
    try:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        with (run / "server.log").open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "scripts/serve_products.py"),
                    "--bundle",
                    str(args.bundle.resolve()),
                    "--port",
                    str(port),
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            base = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 15
            while True:
                if process.poll() is not None:
                    raise RuntimeError("服务启动失败，见server.log")
                try:
                    status, health = request_json(base + "/health")
                    if status == 200 and health["bundle_id"] == service.bundle_id:
                        break
                except (URLError, OSError, TimeoutError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("本地服务未在启动预算内就绪")
                time.sleep(0.05)
            known = next(
                user
                for user in sorted(service.models["bpr"].active_users)
                if len(service.counts.keys() - service.seen.get(user, set())) >= 10
            )
            cold = "__phase2_http_new_user__"
            assert cold not in service.seen
            for name, user, strategy in [
                ("known_mf", known, "mf"),
                ("known_bpr", known, "bpr"),
                ("known_popularity", known, "popularity"),
                ("cold_bpr_popularity", cold, "bpr_cold_popularity"),
            ]:
                expected = service.recommend(user, bundle_id=service.bundle_id, strategy=strategy)
                status, body = request_json(
                    base + "/recommendations",
                    {
                        "user_id": user,
                        "bundle_id": service.bundle_id,
                        "strategy": strategy,
                    },
                )
                assert status == 200 and body == expected, f"HTTP与Python结果不符：{name}"
                report["cases"][name] = {
                    "status": status,
                    "matches_python": True,
                    "n_items": len(body["items"]),
                }
                responses[name] = body
            payload = {"user_id": cold, "bundle_id": "wrong-version", "strategy": "mf"}
            wrong, _ = request_json(base + "/recommendations", payload)
            invalid, _ = request_json(
                base + "/recommendations",
                {**payload, "bundle_id": service.bundle_id, "as_of_ms": 1},
            )
            assert wrong == 409 and invalid == 422
            report.update(
                version_mismatch_status=wrong, arbitrary_time_status=invalid, status="PASS"
            )
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        report["own_server_stopped"] = process is None or process.poll() is not None
        save_json(run / "report.json", report)
        save_json(run / "responses.json", responses)
    print(json.dumps({"run": str(run.resolve()), "report": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
