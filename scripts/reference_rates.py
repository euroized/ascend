"""Reference firing rates from the Shiu et al. 2024 model, for the conditions fixed in docs/PREREG_PHASE0.md.

Each condition builds the network once and runs 30 one-second trials with seeds 0-29. The output
keeps every neuron with a nonzero mean rate, plus per-trial spike counts for the readout neurons.

    python scripts/fetch_data.py && python scripts/fetch_data.py --group provenance
    python scripts/reference_rates.py --workers 2
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ascend import data, hardware, stimuli  # noqa: E402

OUT = data.REPO_ROOT / "results" / "phase0" / "reference_rates.json"
SEEDS = list(range(30))
T_RUN_S = 1.0
FILES = {
    783: ("Completeness_783.csv", "Connectivity_783.parquet"),
    630: ("Completeness_630.csv", "Connectivity_630.parquet"),
}

# Frozen by docs/PREREG_PHASE0.md. Rates in Hz; None means the upstream default (150 Hz).
CONDITIONS = [
    {"name": "v630_sugar_100hz", "version": 630, "stimulus": "sugar", "r_poi_hz": 100},
    {"name": "v630_sugar_200hz", "version": 630, "stimulus": "sugar", "r_poi_hz": 200},
    {"name": "v783_sugar_100hz", "version": 783, "stimulus": "sugar", "r_poi_hz": 100},
    {"name": "v783_sugar_150hz", "version": 783, "stimulus": "sugar", "r_poi_hz": 150},
    {"name": "v783_sugar_200hz", "version": 783, "stimulus": "sugar", "r_poi_hz": 200},
    {"name": "v783_bitter_150hz", "version": 783, "stimulus": "bitter", "r_poi_hz": 150},
    {"name": "v783_jon_all_150hz", "version": 783, "stimulus": "jon_all", "r_poi_hz": 150},
    {"name": "v783_none", "version": 783, "stimulus": "none", "r_poi_hz": None},
]


def run_condition(cond: dict, target: str) -> dict:
    import numpy as np  # noqa: PLC0415

    from ascend import analysis  # noqa: PLC0415
    from ascend.reference import brian2_ref  # noqa: PLC0415

    v = cond["version"]
    path_comp, path_con = (data.path(f) for f in FILES[v])
    idx = stimuli.id_to_index(path_comp)
    fly_ids = np.array(sorted(idx, key=idx.get), dtype=np.int64)
    exc = [idx[i] for i in stimuli.stimulus(cond["stimulus"], v)]
    readouts = stimuli.readouts(v)

    t0 = time.time()
    s = brian2_ref.Session(path_comp, path_con, exc, t_run_s=T_RUN_S, r_poi_hz=cond["r_poi_hz"], target=target)
    s.warmup()
    trials = [s.trial(seed) for seed in SEEDS]
    mean, std = analysis.rates([t.spikes for t in trials], s.n_neurons, T_RUN_S)
    counts = analysis.counts([t.spikes for t in trials], s.n_neurons)

    active = np.flatnonzero(mean > 0)
    print(f"{cond['name']}: {len(active)} active neurons, {time.time() - t0:.0f}s", flush=True)
    return {
        **cond,
        "n_stimulated": len(exc),
        "seeds": SEEDS,
        "t_run_s": T_RUN_S,
        "build_s": round(s.build_s, 2),
        "warmup_s": round(s.warmup_s, 2),
        "run_s": [round(t.run_s, 3) for t in trials],
        "spikes_per_trial": [t.n_spikes for t in trials],
        "readouts": {
            k: {
                "flywire_id": i,
                "rate_hz_mean": round(float(mean[idx[i]]), 4),
                "rate_hz_std": round(float(std[idx[i]]), 4),
                "spike_counts": counts[:, idx[i]].tolist(),
            }
            for k, i in readouts.items()
        },
        "n_active": int(len(active)),
        "rates_hz": {str(fly_ids[i]): [round(float(mean[i]), 4), round(float(std[i]), 4)] for i in active},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=2, help="conditions run in parallel (~3.5 GB RAM each)")
    ap.add_argument("--target", default="cython", help="Brian2 code generation target")
    ap.add_argument("--only", nargs="*", help="run only these condition names")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    conds = [c for c in CONDITIONS if not args.only or c["name"] in args.only]
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = pool.starmap(run_condition, [(c, args.target) for c in conds])

    out = {
        "description": "Reference firing rates, Shiu et al. 2024 model (Brian2), per docs/PREREG_PHASE0.md. "
        "rates_hz maps FlyWire root ID -> [mean, std] over trials, nonzero means only.",
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "workers": args.workers,
        "brian2_target": args.target,
        "hardware": hardware.info(),
        "conditions": {r["name"]: r for r in results},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
