"""Benchmark an engine on the full FlyWire 783 brain and write one JSON result.

    python scripts/fetch_data.py
    python scripts/bench.py --out bench.json

The workload is a stimulus driven by Poisson input (default: 21 sugar GRNs at 150 Hz) for
--duration simulated seconds, repeated --trials times with seeds --seed, --seed+1, ...
The network is built once and warmed up untimed (code generation, spike-queue setup), then
restored to its initial state before every trial.

    real-time factor (RTF) = simulated seconds / wall-clock seconds of the run

RTF >= 1.0 means the simulation keeps pace with real time. The result records the hardware
and software automatically. Submit it with the "Benchmark result" issue template.

--synthetic N runs a small random network instead, for smoke tests; such results are not
valid submissions.
"""

import argparse
import json
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ascend import analysis, data, hardware, stimuli, synthetic  # noqa: E402

SCHEMA = "ascend-bench/1"
ENGINES = ("reference",)


def connectome_info(path_con: Path, entries: list[dict] | None) -> dict:
    import pandas as pd  # noqa: PLC0415

    con = pd.read_parquet(path_con, columns=["Presynaptic_Index", "Postsynaptic_Index", "Connectivity"])
    info = {"connections": int(len(con)), "synapses": int(con["Connectivity"].sum())}
    if entries:
        info["sha256"] = {e["name"]: e["sha256"] for e in entries}
        info["verified"] = all(data.verify(e) for e in entries)
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="reference", choices=ENGINES)
    ap.add_argument("--stimulus", default="sugar", help="stimulus set from data/stimuli_783.json, or 'none'")
    ap.add_argument("--rate-hz", type=float, default=150.0, help="Poisson rate of the stimulated neurons")
    ap.add_argument("--duration", type=float, default=1.0, help="simulated seconds per trial")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="seed of the first trial")
    ap.add_argument("--target", default="cython", help="reference engine: Brian2 code generation target")
    ap.add_argument("--synthetic", type=int, metavar="N", help="use a random N-neuron network (smoke test only)")
    ap.add_argument("--out", type=Path, help="write JSON here (default: print only)")
    args = ap.parse_args()

    from ascend.reference import brian2_ref  # noqa: PLC0415

    tmp = None
    if args.synthetic:
        tmp = tempfile.TemporaryDirectory()
        path_comp, path_con = synthetic.write(Path(tmp.name), n_neurons=args.synthetic,
                                              n_connections=20 * args.synthetic, seed=args.seed)
        exc = list(range(min(10, args.synthetic))) if args.stimulus != "none" else []
        readouts, entries, dataset = {}, None, f"synthetic-{args.synthetic}"
    else:
        entries = data.entries("core")
        path_comp, path_con = data.path("Completeness_783.csv"), data.path("Connectivity_783.parquet")
        idx = stimuli.id_to_index(path_comp)
        exc = [idx[i] for i in stimuli.stimulus(args.stimulus)]
        readouts = {k: idx[i] for k, i in stimuli.readouts().items()}
        dataset = "flywire-783"

    print(f"building {dataset} network ...", flush=True)
    s = brian2_ref.Session(path_comp, path_con, exc, t_run_s=args.duration, r_poi_hz=args.rate_hz,
                           target=args.target)
    s.warmup()
    print(f"build {s.build_s:.1f}s, warmup {s.warmup_s:.1f}s", flush=True)

    seeds = list(range(args.seed, args.seed + args.trials))
    trials = []
    for seed in seeds:
        t = s.trial(seed)
        trials.append(t)
        print(f"trial seed={seed}: run {t.run_s:.2f}s  RTF {t.rtf:.4f}  spikes {t.n_spikes}", flush=True)

    rtf = [t.rtf for t in trials]
    mean, std = analysis.rates([t.spikes for t in trials], s.n_neurons, args.duration)
    med = statistics.median(rtf)
    result = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(["python", *sys.argv]),
        "engine": {"name": args.engine, "implementation": "Shiu et al. 2024 model, Brian2 runtime",
                   "brian2_target": args.target, "dt_ms": 0.1},
        "connectome": {"dataset": dataset, "neurons": s.n_neurons, **connectome_info(path_con, entries)},
        "workload": {"stimulus": args.stimulus, "n_stimulated": len(exc), "rate_hz": args.rate_hz,
                     "t_run_s": args.duration, "trials": args.trials, "seeds": seeds},
        "timing": {"build_s": round(s.build_s, 3), "warmup_s": round(s.warmup_s, 3),
                   "run_s": [round(t.run_s, 4) for t in trials], "rtf": [round(x, 5) for x in rtf],
                   "rtf_median": round(med, 5), "rtf_min": round(min(rtf), 5), "rtf_max": round(max(rtf), 5),
                   "slowdown_vs_realtime": round(1 / med, 2)},
        "fidelity": {"spikes_per_trial": [t.n_spikes for t in trials],
                     "readouts_rate_hz": {k: {"mean": round(float(mean[i]), 3), "std": round(float(std[i]), 3)}
                                          for k, i in readouts.items()}},
        "criteria": {"phase": 0, "note": "Phase 0 baseline, no pass/fail. Phase 1 criteria: docs/PREREG_PHASE1.md."},
        "valid_submission": not args.synthetic,
        "hardware": hardware.info(),
    }
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}")
    print(f"RTF median {med:.4f} ({1 / med:.1f}x slower than real time)")
    if tmp:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
