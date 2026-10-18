"""Readout selectivity of the real connectome against matched null networks (docs/PREREG_BASELINES.md).

    python scripts/baseline_check.py --variant real
    python scripts/baseline_check.py --variant rewired --seed 0     # also weights, random; seeds 0-2
    python scripts/baseline_check.py --summarize --out results/baselines/null_networks.json

Each (variant, seed, condition) is saved to runs/baselines/ when it finishes and skipped on rerun,
so an interrupted run resumes. --summarize combines whatever is saved and scores it.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ascend import baselines, data, fidelity, hardware, stimuli  # noqa: E402
from ascend.engine import CPUEngine, Params  # noqa: E402

RUNS = data.REPO_ROOT / "runs" / "baselines"
REF_CONDITION = {"sugar": "v783_sugar_150hz", "bitter": "v783_bitter_150hz", "jon_all": "v783_jon_all_150hz"}


def run(variant: str, seed: int) -> None:
    p = Params.reference()
    path_comp, path_con = data.path("Completeness_783.csv"), data.path("Connectivity_783.parquet")
    n = len(pd.read_csv(path_comp, index_col=0))
    cols = ["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"]
    con = pd.read_parquet(path_con, columns=cols)
    t0 = time.time()
    net = baselines.network(variant, seed, n, *(con[c].to_numpy() for c in cols), w_syn=p.w_syn)
    del con
    print(f"{variant} seed {seed}: network built in {time.time() - t0:.0f}s", flush=True)

    idx = stimuli.id_to_index(path_comp)
    fly_ids = np.array(sorted(idx, key=idx.get), dtype=np.int64)
    readout_idx = {k: idx[f] for k, f in stimuli.readouts().items()}
    ref = json.loads(fidelity.REFERENCE.read_text())["conditions"]
    n_steps = int(round(baselines.T_RUN_S / p.dt))

    out_dir = RUNS / f"{variant}_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for cond, (stim, rate) in baselines.CONDITIONS.items():
        dest = out_dir / f"{cond}.json"
        if dest.exists():
            continue
        t0 = time.time()
        eng = CPUEngine(net, p, fidelity.stimulated(stim, path_comp), dtype=np.float32)
        counts, wall, runaway = [], [], False
        for s in baselines.SEEDS:
            w0 = time.perf_counter()
            c, runaway = baselines.run_trial(eng, rate, s, n_steps)
            wall.append(time.perf_counter() - w0)
            counts.append(c)
            if runaway:
                break
        counts = np.array(counts)
        rates = counts.mean(0) / baselines.T_RUN_S
        ref_rates = np.zeros(n)
        pos = {int(f): i for i, f in enumerate(fly_ids)}
        for f, (m, _sd) in ref[REF_CONDITION[cond]]["rates_hz"].items():
            ref_rates[pos[int(f)]] = m
        active = (rates > 0) | (ref_rates > 0)
        res = {
            "variant": variant,
            "shuffle_seed": seed,
            "condition": cond,
            "trials": len(counts),
            "runaway": runaway,
            "readouts_hz": {k: round(float(rates[i]), 3) for k, i in readout_idx.items()},
            "groups_hz": {
                g: round(float(np.mean([rates[readout_idx[r]] for r in rs])), 3) for g, rs in baselines.GROUPS.items()
            },
            "spikes_per_trial": float(counts.sum(1).mean()),
            "active_neurons": int((rates > 0).sum()),
            "pearson_vs_reference": round(float(np.corrcoef(rates[active], ref_rates[active])[0, 1]), 4),
            "run_s_median": round(float(np.median(wall)), 3),
        }
        dest.write_text(json.dumps(res) + "\n")
        print(
            f"  {cond}: groups {res['groups_hz']} spikes/trial {res['spikes_per_trial']:.0f} "
            f"r {res['pearson_vs_reference']}{' RUNAWAY' if runaway else ''} ({time.time() - t0:.0f}s)",
            flush=True,
        )


def networks() -> list[tuple[str, int]]:
    return [("real", 0)] + [(v, s) for v in baselines.VARIANTS[1:] for s in baselines.SHUFFLE_SEEDS]


def summarize(out: Path | None) -> int:
    rows, missing = [], []
    for variant, seed in networks():
        d = RUNS / f"{variant}_{seed}"
        conds = {c: d / f"{c}.json" for c in baselines.CONDITIONS}
        if not all(f.exists() for f in conds.values()):
            missing.append(f"{variant}_{seed}")
            continue
        res = {c: json.loads(f.read_text()) for c, f in conds.items()}
        sc = baselines.score({c: r["groups_hz"] for c, r in res.items()}, {c: r["runaway"] for c, r in res.items()})
        rows.append({"variant": variant, "shuffle_seed": seed, **sc, "conditions": res})
        print(f"{variant:8s} seed {seed}: score {sc['score']}/{sc['of']}")
    if missing:
        print(f"missing: {', '.join(missing)}")
        return 2
    real = [r for r in rows if r["variant"] == "real"]
    nulls = [r for r in rows if r["variant"] != "real"]
    c1 = all(r["score"] == r["of"] for r in real)
    c2 = all(r["score"] < r["of"] for r in nulls)
    result = {
        "schema": "ascend-baselines/1",
        "prereg": "docs/PREREG_BASELINES.md",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine": {"device": "cpu", "dtype": "float32"},
        "seeds": baselines.SEEDS,
        "thresholds_hz": {"on": baselines.ON_HZ, "off": baselines.OFF_HZ},
        "criteria": {"C1_real_scores_6_of_6": c1, "C2_every_null_below_6": c2, "claim": c1 and c2},
        "networks": rows,
        "hardware": hardware.info(),
    }
    print(f"C1 {c1}  C2 {c2}  claim {c1 and c2}")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=baselines.VARIANTS)
    ap.add_argument("--seed", type=int, default=0, choices=baselines.SHUFFLE_SEEDS)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.summarize:
        return summarize(args.out)
    if args.variant is None:
        ap.error("give --variant, or --summarize")
    run(args.variant, 0 if args.variant == "real" else args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
