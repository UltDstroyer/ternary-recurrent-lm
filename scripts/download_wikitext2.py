"""Download and verify the fixed WikiText-2 splits used by the experiment."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

BASE_URL = (
    "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2"
)
FILES = {
    "train.txt": "9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f",
    "valid.txt": "f0737ed31fc1329026e95cb8b98e19c2a182c39c240ab909dc31abf2f8af58e8",
    "test.txt": "d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0",
}


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    destination = Path("data/external/wikitext-2")
    destination.mkdir(parents=True, exist_ok=True)
    for filename, expected in FILES.items():
        target = destination / filename
        if not target.exists() or checksum(target) != expected:
            temporary = target.with_suffix(".download")
            urllib.request.urlretrieve(f"{BASE_URL}/{filename}", temporary)
            temporary.replace(target)
        actual = checksum(target)
        if actual != expected:
            raise RuntimeError(f"checksum mismatch for {target}: {actual}")
        print(f"verified {target} ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
