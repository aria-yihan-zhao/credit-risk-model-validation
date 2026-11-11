"""Download and verify the official UCI source workbook."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
EXPECTED_SHA256 = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"
ARCHIVE_MEMBER = "default of credit card clients.xls"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT_ROOT / "data" / "raw" / ARCHIVE_MEMBER


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    if DESTINATION.exists():
        print(f"Source workbook already exists: {DESTINATION}")
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        archive = Path(temp_dir) / "uci_credit_default.zip"
        print(f"Downloading official source from {URL}")
        urllib.request.urlretrieve(URL, archive)
        actual_hash = sha256(archive)
        if actual_hash != EXPECTED_SHA256:
            raise RuntimeError(
                f"Archive checksum mismatch: expected {EXPECTED_SHA256}, got {actual_hash}"
            )
        with zipfile.ZipFile(archive) as bundle:
            if ARCHIVE_MEMBER not in bundle.namelist():
                raise RuntimeError(f"Expected workbook missing from archive: {ARCHIVE_MEMBER}")
            with bundle.open(ARCHIVE_MEMBER) as source, DESTINATION.open("wb") as target:
                shutil.copyfileobj(source, target)
    print(f"Verified and extracted: {DESTINATION}")


if __name__ == "__main__":
    main()
