import hashlib

import pytest

from ascend import data


def _entry(src, content):
    return {"name": "f.bin", "url": src.as_uri(), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def test_fetch_verifies_and_skips(tmp_path):
    content = b"connectome" * 1000
    src = tmp_path / "src.bin"
    src.write_bytes(content)
    e = _entry(src, content)
    dest = data.fetch(e, tmp_path / "out")
    assert dest.read_bytes() == content and data.verify(e, tmp_path / "out")
    src.unlink()  # already verified: no download attempted
    assert data.fetch(e, tmp_path / "out") == dest


def test_fetch_rejects_bad_checksum(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"tampered")
    e = _entry(src, b"original")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        data.fetch(e, tmp_path / "out", retries=0)
    assert not (tmp_path / "out" / "f.bin").exists()
