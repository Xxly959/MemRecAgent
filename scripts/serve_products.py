"""仅在127.0.0.1启动已验证冻结包的推荐API，请求不触发训练。"""

import argparse
from pathlib import Path

import torch
import uvicorn

from memrecagent.product_api import create_app
from memrecagent.product_bundle import load_bundle


def main() -> None:
    """先完成包校验和CPU加载，再绑定本地端口；不自动挑选最新包。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port必须为1至65535")
    torch.set_num_threads(1)
    service = load_bundle(args.bundle)
    uvicorn.run(create_app(service), host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
