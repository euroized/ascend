"""Locate, fetch and verify the pinned input files listed in data/manifest.json."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "data" / "manifest.json"


def data_dir() -> Path:
    """Directory holding fetched data. Override with the ASCEND_DATA environment variable."""
    return Path(os.environ.get("ASCEND_DATA", REPO_ROOT / "data"))


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def entries(group: str = "core") -> list[dict]:
    return load_manifest()["groups"][group]


def path(name: str) -> Path:
    """Path of a manifest file inside data_dir(). Raises if it has not been fetched."""
    p = data_dir() / name
    if not p.is_file():
        raise FileNotFoundError(f"{p} not found. Run: python scripts/fetch_data.py")
    return p


def sha256(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify(entry: dict, directory: Path | None = None) -> bool:
    p = (directory or data_dir()) / entry["name"]
    return p.is_file() and p.stat().st_size == entry["bytes"] and sha256(p) == entry["sha256"]


def fetch(entry: dict, directory: Path | None = None, retries: int = 4) -> Path:
    """Download one manifest entry, verify it, and return its path. Skips files already verified."""
    directory = directory or data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / entry["name"]
    if verify(entry, directory):
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(entry["url"], timeout=60) as r, open(tmp, "wb") as f:
                while block := r.read(1 << 20):
                    f.write(block)
            break
        except OSError:
            if attempt == retries:
                raise
            time.sleep(2 ** (attempt + 1))
    got = sha256(tmp)
    if got != entry["sha256"]:
        tmp.unlink()
        raise ValueError(f"SHA-256 mismatch for {entry['name']}: expected {entry['sha256']}, got {got}")
    tmp.replace(dest)
    return dest
