"""Exact-tier check: does the ASCEND engine compute the reference model spike for spike?

Draws one set of Poisson stimulus kicks (Bernoulli(rate * dt) per stimulated neuron and step,
seeded), feeds the identical kicks to the Brian2 reference network (ascend.reference.brian2_ref
.run_events) and to the ASCEND float64 CPU engine, and compares every spike.

    python scripts/fetch_data.py
    python scripts/exact_check.py --stimulus sugar --duration 1.0 --out exact.json
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ascend import data, hardware, stimuli  # noqa: E402
from ascend.engine import CPUEngine, EventInput, Network, Params  # noqa: E402
from ascend.reference import brian2_ref  # noqa: E402


def compare(ours: dict, ref: dict, dt: float) -> dict:
    """Spike-by-spike comparison of two {neuron: times} dicts, on integer time steps."""
    steps = {k: np.rint(np.asarray(v) / dt).astype(np.int64) for k, v in ref.items()}
    mine = {k: np.rint(np.asarray(v) / dt).astype(np.int64) for k, v in ours.items()}
    neurons = sorted(set(steps) | set(mine))
    differing = [k for k in neurons if not np.array_equal(steps.get(k, []), mine.get(k, []))]
    first = None
    for k in differing:
        a, b = mine.get(k, np.array([], int)), steps.get(k, np.array([], int))
        m = min(len(a), len(b))
        idx = int(np.flatnonzero(a[:m] != b[:m])[0]) if np.any(a[:m] != b[:m]) else m
        t = min(a[idx] if idx < len(a) else 1 << 62, b[idx] if idx < len(b) else 1 << 62)
        if first is None or t < first["step"]:
            first = {"neuron": int(k), "step": int(t)}
    return {
        "spikes_engine": int(sum(len(v) for v in mine.values())),
        "spikes_reference": int(sum(len(v) for v in steps.values())),
        "neurons_active_engine": len(mine),
        "neurons_active_reference": len(steps),
        "neurons_differing": len(differing),
        "first_difference": first,
        "identical": not differing,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stimulus", default="sugar")
    ap.add_argument("--rate-hz", type=float, default=150.0)
    ap.add_argument("--duration", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", default="cython", help="Brian2 code generation target for the reference")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    p = Params.reference()
    n_steps = int(round(args.duration / p.dt))
    path_comp, path_con = data.path("Completeness_783.csv"), data.path("Connectivity_783.parquet")
    idx = stimuli.id_to_index(path_comp)
    exc = sorted(idx[i] for i in stimuli.stimulus(args.stimulus))
    rng = np.random.default_rng(args.seed)
    kick_steps, kick_pos = np.nonzero(rng.random((n_steps, len(exc))) < args.rate_hz * p.dt)

    t0 = time.perf_counter()
    eng = CPUEngine(Network.from_files(path_comp, path_con, p.w_syn), p, exc)
    t1 = time.perf_counter()
    ours = eng.run(n_steps, EventInput(kick_steps, kick_pos)).trains()
    t2 = time.perf_counter()
    ref = brian2_ref.run_events(path_comp, path_con, exc, kick_steps, kick_pos, n_steps=n_steps, target=args.target)
    t3 = time.perf_counter()

    result = {
        "schema": "ascend-exact/1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(["python", *sys.argv]),
        "engine": {"name": "ascend-cpu", "dtype": "float64"},
        "reference": {"name": "Shiu et al. 2024 model, Brian2", "brian2_target": args.target},
        "workload": {
            "stimulus": args.stimulus,
            "n_stimulated": len(exc),
            "rate_hz": args.rate_hz,
            "n_steps": n_steps,
            "kicks": int(len(kick_steps)),
            "seed": args.seed,
        },
        "comparison": compare(ours, ref, p.dt),
        "timing_s": {
            "engine_build": round(t1 - t0, 2),
            "engine_run": round(t2 - t1, 2),
            "reference_build_and_run": round(t3 - t2, 2),
        },
        "hardware": hardware.info(),
    }
    print(json.dumps(result["comparison"], indent=1))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    return 0 if result["comparison"]["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
