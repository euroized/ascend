"""The reference model must stay byte-identical to upstream model.py @ 91bdd1e."""

from ascend import data

UPSTREAM_SHA256 = "fc45837d7122c6ce2a7f3f2f23c515992e4b232aadb919efabb72337fac88e4e"


def test_shiu_model_is_verbatim():
    assert data.sha256(data.REPO_ROOT / "ascend" / "reference" / "shiu_model.py") == UPSTREAM_SHA256


def test_shiu_license_is_kept():
    text = (data.REPO_ROOT / "ascend" / "reference" / "LICENSE.shiu").read_text()
    assert "MIT License" in text and "Philip Shiu" in text
