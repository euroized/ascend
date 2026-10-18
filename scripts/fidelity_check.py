"""Run the preregistered statistical fidelity suite (docs/PREREG_PHASE1.md) on one engine variant.

Variants (the controls fixed by the preregistration):
    exact     the engine as built                                  (positive control: must pass)
    delay17   synaptic delay 17 steps instead of 18                (negative control: must fail)
    weight95  every synaptic weight x 0.95                         (negative control: must fail)

Each condition is saved to runs/phase1_fidelity/<name>/ when it finishes and skipped on rerun,
so an interrupted run resumes. The combined result is written once all conditions are done.

    python scripts/fidelity_check.py --device cpu --dtype float64 --variant exact --out f.json
"""

import argparse
import dataclasses
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ascend import data, fidelity, hardware, stimuli  # noqa: E402
from ascend.engine import Network, Params, make  # noqa: E402

VARIANTS = ("exact", "delay17", "weight95")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda", "emulate"))
    ap.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    ap.add_argument("--variant", default="exact", choices=VARIANTS)
    ap.add_argument("--only", nargs="*", help="run only these conditions")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    name = f"{args.device}_{args.dtype}_{args.variant}"
    partial = data.REPO_ROOT / "runs" / "phase1_fidelity" / name
    partial.mkdir(parents=True, exist_ok=True)

    p = Params.reference()
    if args.variant == "delay17":
        p = dataclasses.replace(p, t_dly=17 * p.dt)
    path_comp, path_con = data.path("Completeness_783.csv"), data.path("Connectivity_783.parquet")
    net = Network.from_files(path_comp, path_con, p.w_syn)
    if args.variant == "weight95":
        net = dataclasses.replace(net, weight=net.weight * 0.95)
    dtype = {"float32": np.float32, "float64": np.float64}[args.dtype]
    idx = stimuli.id_to_index(path_comp)
    fly_ids = np.array(sorted(idx, key=idx.get), dtype=np.int64)
    ref = json.loads(fidelity.REFERENCE.read_text())["conditions"]

    for cond, (stim, rate) in fidelity.CONDITIONS.items():
        if (args.only and cond not in args.only) or (partial / f"{cond}.json").exists():
            continue
        t0 = time.time()
        eng_factory = lambda st: make(net, p, fidelity.stimulated(st, path_comp), device=args.device, dtype=dtype)  # noqa: E731
        r = fidelity.run_condition(eng_factory, stim, rate, net.n_neurons, p.dt)
        res = {**fidelity.compare(r["counts"], ref[cond], fly_ids), "run_s_median": float(np.median(r["run_s"]))}
        (partial / f"{cond}.json").write_text(json.dumps(res) + "\n")
        ok = all(c["pass"] for c in res["checks"])
        print(f"{cond}: {'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f}s)", flush=True)

    done = {
        c: json.loads((partial / f"{c}.json").read_text())
        for c in fidelity.CONDITIONS
        if (partial / f"{c}.json").exists()
    }
    if len(done) < len(fidelity.CONDITIONS):
        print(f"{len(done)}/{len(fidelity.CONDITIONS)} conditions done; rerun to continue")
        return 2
    passed = all(c["pass"] for r in done.values() for c in r["checks"])
    result = {
        "schema": "ascend-fidelity/1",
        "prereg": "docs/PREREG_PHASE1.md",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine": {"device": args.device, "dtype": args.dtype, "variant": args.variant},
        "seeds": fidelity.SEEDS,
        "pass": passed,
        "conditions": done,
        "hardware": hardware.info(),
    }
    for cond, r in done.items():
        for c in r["checks"]:
            print(f"[{'PASS' if c['pass'] else 'FAIL'}] {cond} {c['check']}")
    print(f"{name}: {'PASS' if passed else 'FAIL'}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
