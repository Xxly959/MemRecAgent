"""Download only the official MovieLens 100K ratings and research use terms."""

import argparse
import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

BASE = "https://files.grouplens.org/datasets/movielens/ml-100k/"
RATINGS_SHA256 = "06416e597f82b7342361e41163890c81036900f418ad91315590814211dca490"


def main() -> None:
    """Keep cached files, verify ratings bytes, and preserve first provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination", type=Path, default=Path(__file__).resolve().parents[1] / "data/raw/ml-100k"
    )
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ["README", "u.data"]:
        target = args.destination / name
        if target.exists():
            data = target.read_bytes()
        else:
            with urlopen(BASE + name, timeout=40) as response:
                data = response.read(5_000_001)
            if not data or len(data) > 5_000_000:
                raise ValueError(f"Unexpected download size for {name}")
            if name == "u.data" and sha256(data).hexdigest() != RATINGS_SHA256:
                raise ValueError("Official ratings bytes changed; inspect before updating checksum")
            with target.open("xb") as stream:
                stream.write(data)
        digest = sha256(data).hexdigest()
        if name == "u.data" and digest != RATINGS_SHA256:
            raise ValueError("Cached ratings checksum mismatch; existing data preserved")
        files[name] = {"url": BASE + name, "sha256": digest, "bytes": len(data)}
        print(name, files[name])
    manifest_path = args.destination / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["files"] != files:
            raise ValueError("Existing provenance does not match files; nothing overwritten")
    else:
        manifest = {
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
            "dataset": "MovieLens 100K",
            "files": files,
            "license_source": BASE + "README",
            "purpose": "Local research and learning; do not redistribute raw data.",
        }
        with manifest_path.open("x") as stream:
            json.dump(manifest, stream, indent=2)


if __name__ == "__main__":
    main()
