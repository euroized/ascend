"""Check a community bench.json against the Phase 1 preregistration (docs/PREREG_PHASE1.md, Amendment 1).

A submission is *eligible* if it is an unmodified, full-connectome CUDA run of the preregistered
workload. An eligible submission meets *claim rule 3* if, on a consumer GPU, S1 (median RTF >= 1),
F (fidelity suite) and G (same-input tier) all pass. Rule 4 (an independent reproduction: a second
rule-3 submission from a different person on a different GPU model) is evaluated across
submissions by `rule4`.

Every value is re-derived from the raw fields of the JSON where it can be; the JSON's own
`criteria` block is not trusted.
"""

from __future__ import annotations

import re
import statistics
import subprocess

from . import data, fidelity, sameinput

SCHEMA = "ascend-bench/2"
WORKLOAD = {"stimulus": "sugar", "rate_hz": 150.0, "t_run_s": 1.0, "trials": 5, "seeds": [0, 1, 2, 3, 4]}
RTF_TARGET = 1.0
CONSUMER = re.compile(r"\b(geforce|titan)\b", re.I)
# Data-centre and professional workstation cards (PREREG_PHASE1.md, S tier): published but not counted.
NON_CONSUMER = re.compile(
    r"\b(quadro|tesla|rtx a\d+|rtx pro|ada generation|[abghlt]\d{1,3}s?|gh200|gb200)\b",
    re.I,
)


def gpu_name(sub: dict) -> str | None:
    gpus = sub.get("hardware", {}).get("gpus") or []
    return gpus[0].get("name") if gpus else None


def is_consumer(name: str | None) -> bool:
    n = name or ""
    return bool(CONSUMER.search(n)) and not NON_CONSUMER.search(n)


def _commit_known(commit: str | None) -> bool | None:
    """True/False if the commit is/isn't in the local repository history; None if git is unavailable."""
    if not commit:
        return False
    try:
        r = subprocess.run(
            ["git", "-C", str(data.REPO_ROOT), "cat-file", "-e", f"{commit}^{{commit}}"],
            capture_output=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return r.returncode == 0


def check(sub: dict) -> dict:
    checks = []

    def add(group: str, name: str, ok, detail=None):
        checks.append({"group": group, "check": name, "pass": None if ok is None else bool(ok), "detail": detail})

    manifest = data.load_manifest()
    con, eng, wl = sub.get("connectome", {}), sub.get("engine", {}), sub.get("workload", {})
    repo = sub.get("hardware", {}).get("repo", {})

    # Integrity: an unmodified full-connectome CUDA run of the preregistered workload.
    add("integrity", "schema is ascend-bench/2", sub.get("schema") == SCHEMA, sub.get("schema"))
    add("integrity", "valid_submission is true", sub.get("valid_submission") is True)
    add("integrity", "unmodified checkout (repo.dirty is false)", repo.get("dirty") is False, repo.get("dirty"))
    add("integrity", "commit is in this repository's history", _commit_known(repo.get("commit")), repo.get("commit"))
    expect_sha = {e["name"]: e["sha256"] for e in manifest["groups"]["core"]}
    add(
        "integrity",
        "connectome files match data/manifest.json",
        con.get("sha256") == expect_sha and con.get("verified") is True,
        con.get("dataset"),
    )
    counts = (con.get("neurons"), con.get("connections"), con.get("synapses"))
    m = manifest["connectome"]
    add(
        "integrity",
        "connectome size is FlyWire v783",
        counts == (m["neurons"], m["connections"], m["synapses"]),
        counts,
    )
    add(
        "integrity",
        "engine is ascend on cuda in float32",
        (eng.get("name"), eng.get("device"), eng.get("dtype")) == ("ascend", "cuda", "float32"),
        f"{eng.get('name')}/{eng.get('device')}/{eng.get('dtype')}",
    )
    add(
        "integrity",
        "workload is the preregistered default",
        all(wl.get(k) == v for k, v in WORKLOAD.items()),
        {k: wl.get(k) for k in WORKLOAD},
    )

    # S1, recomputed from the per-trial run times.
    run_s = sub.get("timing", {}).get("run_s") or []
    rtf = [WORKLOAD["t_run_s"] / r for r in run_s if r and r > 0]
    med = statistics.median(rtf) if len(rtf) == WORKLOAD["trials"] else None
    add(
        "S",
        "median RTF >= 1.0 (recomputed from run_s)",
        None if med is None else med >= RTF_TARGET,
        None if med is None else round(med, 4),
    )

    # F: present, preregistered conditions and seeds, every check passed.
    fs = sub.get("fidelity_suite") or {}
    conds = fs.get("conditions") or {}
    f_ok = (
        bool(fs)
        and list(conds) == list(fidelity.CONDITIONS)
        and fs.get("seeds") == fidelity.SEEDS
        and all(c.get("pass") is True for r in conds.values() for c in r.get("checks", []))
        and all(r.get("checks") for r in conds.values())
    )
    failed = [f"{k}: {c['check']}" for k, r in conds.items() for c in r.get("checks", []) if not c.get("pass")]
    add("F", "fidelity suite present and every check passes", f_ok if fs else None, failed or None)

    # G: present, preregistered settings, Jaccard >= 0.99 in every workload (recomputed).
    g = sub.get("same_input") or {}
    w = g.get("workloads") or {}
    g_ok = (
        bool(g)
        and g.get("kick_seed") == sameinput.KICK_SEED
        and set(w) == set(sameinput.WORKLOADS)
        and all(r.get("jaccard", 0) >= sameinput.MIN_JACCARD for r in w.values())
    )
    add(
        "G",
        "same-input Jaccard >= 0.99 in all workloads",
        g_ok if g else None,
        {k: r.get("jaccard") for k, r in w.items()} or None,
    )

    name = gpu_name(sub)
    consumer = is_consumer(name)
    add("hardware", "consumer GPU (GeForce or TITAN)", consumer, name)

    def group_ok(gr):
        return all(c["pass"] is True for c in checks if c["group"] == gr)

    eligible = group_ok("integrity")
    rule3 = eligible and consumer and all(group_ok(gr) for gr in ("S", "F", "G"))
    return {
        "gpu": name,
        "driver": (sub.get("hardware", {}).get("gpus") or [{}])[0].get("driver"),
        "consumer_gpu": consumer,
        "rtf_median": None if med is None else round(med, 4),
        "eligible": eligible,
        "S1": group_ok("S"),
        "F": group_ok("F"),
        "G": group_ok("G"),
        "rule3": rule3,
        "checks": checks,
    }


def rule4(verdicts: list[dict]) -> dict:
    """Rule 3 met by at least two submissions from different submitters on different GPU models."""
    passing = [v for v in verdicts if v["verdict"]["rule3"]]
    for i, a in enumerate(passing):
        for b in passing[i + 1 :]:
            if a["submitter"] != b["submitter"] and a["verdict"]["gpu"] != b["verdict"]["gpu"]:
                return {"met": True, "pair": [a["id"], b["id"]]}
    return {"met": False, "rule3_submissions": [v["id"] for v in passing]}
