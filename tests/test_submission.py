"""Community submission checker (ascend/submission.py, scripts/check_submission.py)."""

import importlib.util
import json
import subprocess

import pytest

from ascend import data, fidelity, sameinput, submission


def head_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(data.REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def make_bench(gpu="NVIDIA GeForce RTX 4070", run_s=0.1) -> dict:
    m = data.load_manifest()
    return {
        "schema": "ascend-bench/2",
        "engine": {"name": "ascend", "device": "cuda", "dtype": "float32", "dt_ms": 0.1},
        "connectome": {
            "dataset": "flywire-783",
            "neurons": m["connectome"]["neurons"],
            "connections": m["connectome"]["connections"],
            "synapses": m["connectome"]["synapses"],
            "sha256": {e["name"]: e["sha256"] for e in m["groups"]["core"]},
            "verified": True,
        },
        "workload": {
            "stimulus": "sugar",
            "n_stimulated": 21,
            "rate_hz": 150.0,
            "t_run_s": 1.0,
            "trials": 5,
            "seeds": [0, 1, 2, 3, 4],
        },
        "timing": {"run_s": [run_s] * 5, "rtf_median": 1 / run_s},
        "fidelity_suite": {
            "seeds": list(fidelity.SEEDS),
            "pass": True,
            "conditions": {c: {"checks": [{"check": "x", "pass": True}]} for c in fidelity.CONDITIONS},
        },
        "same_input": {
            "kick_seed": sameinput.KICK_SEED,
            "pass": True,
            "workloads": {w: {"jaccard": 1.0, "pass": True} for w in sameinput.WORKLOADS},
        },
        "criteria": {"S1_rtf_median_ge_1": True, "F_fidelity_suite": True, "G_same_input": True},
        "valid_submission": True,
        "hardware": {
            "gpus": [{"name": gpu, "driver": "560.35"}],
            "repo": {"commit": head_commit(), "dirty": False},
        },
    }


def test_valid_submission_meets_rule3():
    v = submission.check(make_bench())
    assert v["eligible"] and v["S1"] and v["F"] and v["G"] and v["consumer_gpu"] and v["rule3"]
    assert v["rtf_median"] == 10.0 and v["gpu"] == "NVIDIA GeForce RTX 4070" and v["driver"] == "560.35"
    json.dumps(v)


def _set(d, path, value):
    *head, last = path
    for k in head:
        d = d[k]
    if value is KeyError:
        del d[last]
    else:
        d[last] = value


@pytest.mark.parametrize(
    "path,value,flag",
    [
        (("hardware", "repo", "dirty"), True, "eligible"),
        (("hardware", "repo", "commit"), "0" * 40, "eligible"),
        (("valid_submission",), False, "eligible"),
        (("connectome", "verified"), False, "eligible"),
        (("connectome", "synapses"), 1000, "eligible"),
        (("engine", "dtype"), "float64", "eligible"),
        (("engine", "device"), "cpu", "eligible"),
        (("workload", "stimulus"), "bitter", "eligible"),
        (("timing", "run_s"), [1.5] * 5, "S1"),
        (("timing", "run_s"), [0.1] * 4, "S1"),
        (("fidelity_suite",), None, "F"),
        (("fidelity_suite", "seeds"), list(range(10)), "F"),
        (("fidelity_suite", "conditions", "v783_none", "checks"), [{"check": "x", "pass": False}], "F"),
        (("fidelity_suite", "conditions", "v783_none"), KeyError, "F"),
        (("same_input",), None, "G"),
        (("same_input", "kick_seed"), 0, "G"),
        (("same_input", "workloads", "bitter", "jaccard"), 0.98, "G"),
        (("same_input", "workloads", "jon_all"), KeyError, "G"),
    ],
)
def test_each_failure_blocks_rule3(path, value, flag):
    b = make_bench()
    _set(b, path, value)
    v = submission.check(b)
    assert v[flag] is False and v["rule3"] is False


def test_self_reported_criteria_are_not_trusted():
    b = make_bench(run_s=2.0)
    b["timing"]["rtf_median"] = 50.0
    assert submission.check(b)["S1"] is False


@pytest.mark.parametrize(
    "name,consumer",
    [
        ("NVIDIA GeForce RTX 4070", True),
        ("NVIDIA GeForce RTX 3060 Laptop GPU", True),
        ("NVIDIA GeForce GTX 1080 Ti", True),
        ("NVIDIA TITAN RTX", True),
        ("NVIDIA A100-SXM4-80GB", False),
        ("NVIDIA H100 PCIe", False),
        ("Tesla T4", False),
        ("NVIDIA L4", False),
        ("NVIDIA L40S", False),
        ("NVIDIA RTX A4000", False),
        ("NVIDIA RTX 6000 Ada Generation", False),
        ("Quadro RTX 5000", False),
        ("NVIDIA RTX PRO 6000 Blackwell", False),
        (None, False),
    ],
)
def test_consumer_gpu(name, consumer):
    assert submission.is_consumer(name) is consumer


def test_data_centre_gpu_is_eligible_but_not_rule3():
    v = submission.check(make_bench(gpu="NVIDIA A100-SXM4-80GB"))
    assert v["eligible"] and v["S1"] and not v["consumer_gpu"] and not v["rule3"]


def _rec(id_, who, gpu, ok=True):
    return {"id": id_, "submitter": who, "verdict": {"rule3": ok, "gpu": gpu}}


def test_rule4_needs_different_person_and_gpu():
    a = _rec("1-alice", "alice", "NVIDIA GeForce RTX 4070")
    assert not submission.rule4([a, _rec("2-alice", "alice", "NVIDIA GeForce RTX 3060")])["met"]
    assert not submission.rule4([a, _rec("3-bob", "bob", "NVIDIA GeForce RTX 4070")])["met"]
    assert not submission.rule4([a, _rec("4-bob", "bob", "NVIDIA GeForce RTX 3060", ok=False)])["met"]
    r = submission.rule4([a, _rec("5-bob", "bob", "NVIDIA GeForce RTX 3060")])
    assert r == {"met": True, "pair": ["1-alice", "5-bob"]}


@pytest.fixture
def script(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("check_submission", data.REPO_ROOT / "scripts/check_submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "BOARD", tmp_path / "community")
    return mod


def test_add_and_board(script, tmp_path, monkeypatch):
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps(make_bench()))
    monkeypatch.setattr("sys.argv", ["x", str(bench), "--issue", "12", "--submitter", "octocat", "--add"])
    assert script.main() == 0
    stored = json.loads((script.BOARD / "12-octocat.json").read_text())
    assert stored["issue"] == 12 and stored["submitter"] == "octocat" and stored["bench"] == make_bench()
    board = (script.BOARD / "README.md").read_text()
    assert "@octocat" in board and "NVIDIA GeForce RTX 4070" in board and "10.0" in board
    assert "not met" in board


def test_failing_submission_exits_nonzero(script, tmp_path, monkeypatch):
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps(make_bench(run_s=2.0)))
    monkeypatch.setattr("sys.argv", ["x", str(bench)])
    assert script.main() == 1
    assert not list(script.BOARD.glob("*.json"))


def test_committed_board_is_current(script, monkeypatch):
    monkeypatch.setattr(script, "BOARD", data.REPO_ROOT / "results" / "community")
    assert script.render_board() == (script.BOARD / "README.md").read_text()
