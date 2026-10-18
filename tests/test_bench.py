import json
import subprocess
import sys

from ascend import data


def test_bench_synthetic_writes_valid_json(tmp_path):
    out = tmp_path / "bench.json"
    cmd = [sys.executable, str(data.REPO_ROOT / "scripts" / "bench.py"), "--synthetic", "150",
           "--trials", "2", "--duration", "0.05", "--target", "numpy", "--out", str(out)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=600)
    r = json.loads(out.read_text())
    assert r["schema"] == "ascend-bench/1"
    assert r["valid_submission"] is False
    assert r["workload"]["seeds"] == [0, 1]
    assert len(r["timing"]["rtf"]) == 2 and r["timing"]["rtf_median"] > 0
    assert r["hardware"]["cpu"]
