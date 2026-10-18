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

Engines:
    --engine reference                  Shiu et al. 2024 model in Brian2 (Phase 0 baseline)
    --engine ascend --device cuda       ASCEND engine on an NVIDIA GPU (needs CuPy)
    --engine ascend --device cpu        ASCEND engine on the CPU (float64, exact)

--fidelity also runs the preregistered statistical fidelity suite (docs/PREREG_PHASE1.md):
30 trials of each reference condition, compared with results/phase0/reference_rates.json.
A Phase 1 submission is

    python scripts/bench.py --engine ascend --device cuda --fidelity --out bench.json

--synthetic N runs a small random network instead, for smoke tests; such results are not
valid submissions.
"""

import argparse
import json
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from ascend import analysis, data, hardware, stimuli, synthetic  # noqa: E402

SCHEMA = "ascend-bench/2"
ENGINES = ("reference", "ascend")
RTF_TARGET = 1.0


class ReferenceRunner:
    """Brian2 reference model: build once, store/restore per trial."""

    def __init__(self, path_comp, path_con, exc, args):
        from ascend.reference import brian2_ref  # noqa: PLC0415

        self.s = brian2_ref.Session(
            path_comp, path_con, exc, t_run_s=args.duration, r_poi_hz=args.rate_hz, target=args.target
        )
        self.n_neurons, self.build_s = self.s.n_neurons, self.s.build_s
        self.info = {"implementation": "Shiu et al. 2024 model, Brian2 runtime", "brian2_target": args.target}

    def warmup(self) -> float:
        return self.s.warmup()

    def trial(self, seed: int) -> tuple[float, dict]:
        t = self.s.trial(seed)
        return t.run_s, t.spikes


class AscendRunner:
    """ASCEND engine on --device; the network is built once, the engine reset before each trial."""

    def __init__(self, path_comp, path_con, exc, args):
        from ascend import engine  # noqa: PLC0415

        t0 = time.perf_counter()
        self.params = engine.Params.reference()
        self.net = engine.Network.from_files(path_comp, path_con, self.params.w_syn)
        dtype = {"float32": np.float32, "float64": np.float64}.get(args.dtype)
        self.make = lambda stim: engine.make(self.net, self.params, stim, device=args.device, dtype=dtype)
        self.eng = self.make(exc)
        self.build_s = time.perf_counter() - t0
        self.n_neurons, self.args = self.net.n_neurons, args
        self.n_steps = int(round(args.duration / self.params.dt))
        self.info = {
            "implementation": "ASCEND event-driven engine",
            "device": args.device,
            "dtype": str(self.eng.dtype),
        }

    def warmup(self) -> float:
        t0 = time.perf_counter()
        if hasattr(self.eng, "warmup"):
            self.eng.warmup(self.n_steps)
        return time.perf_counter() - t0

    def trial(self, seed: int) -> tuple[float, dict]:
        from ascend.engine import PoissonInput  # noqa: PLC0415

        self.eng.reset()
        t0 = time.perf_counter()
        s = self.eng.run(self.n_steps, PoissonInput(self.args.rate_hz, seed))
        return time.perf_counter() - t0, s.trains()


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
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu", "emulate"), help="ascend engine device")
    ap.add_argument("--dtype", choices=("float32", "float64"), help="ascend engine precision (default per device)")
    ap.add_argument("--fidelity", action="store_true", help="also run the preregistered fidelity suite")
    ap.add_argument("--synthetic", type=int, metavar="N", help="use a random N-neuron network (smoke test only)")
    ap.add_argument("--out", type=Path, help="write JSON here (default: print only)")
    args = ap.parse_args()

    tmp = None
    if args.synthetic:
        tmp = tempfile.TemporaryDirectory()
        path_comp, path_con = synthetic.write(
            Path(tmp.name), n_neurons=args.synthetic, n_connections=20 * args.synthetic, seed=args.seed
        )
        exc = list(range(min(10, args.synthetic))) if args.stimulus != "none" else []
        readouts, entries, dataset = {}, None, f"synthetic-{args.synthetic}"
    else:
        entries = data.entries("core")
        path_comp, path_con = data.path("Completeness_783.csv"), data.path("Connectivity_783.parquet")
        idx = stimuli.id_to_index(path_comp)
        exc = [idx[i] for i in stimuli.stimulus(args.stimulus)]
        readouts = {k: idx[i] for k, i in stimuli.readouts().items()}
        dataset = "flywire-783"

    selftest = None
    if args.engine == "ascend" and args.device != "cpu":
        from ascend.engine import selftest as st  # noqa: PLC0415

        selftest = st.run(args.device)
        print(f"self-test vs CPU engine: {'OK' if selftest['ok'] else 'MISMATCH'} {selftest}", flush=True)

    print(f"building {dataset} network for engine {args.engine} ...", flush=True)
    runner = (ReferenceRunner if args.engine == "reference" else AscendRunner)(path_comp, path_con, exc, args)
    warmup_s = runner.warmup()
    print(f"build {runner.build_s:.1f}s, warmup {warmup_s:.1f}s", flush=True)

    seeds = list(range(args.seed, args.seed + args.trials))
    run_s, spikes = [], []
    for seed in seeds:
        r, sp = runner.trial(seed)
        run_s.append(r)
        spikes.append(sp)
        n = sum(len(v) for v in sp.values())
        print(f"trial seed={seed}: run {r:.3f}s  RTF {args.duration / r:.4f}  spikes {n}", flush=True)

    rtf = [args.duration / r for r in run_s]
    mean, std = analysis.rates(spikes, runner.n_neurons, args.duration)
    med = statistics.median(rtf)

    fidelity = None
    if args.fidelity:
        if args.engine != "ascend" or args.synthetic:
            raise SystemExit("--fidelity needs --engine ascend on the full connectome")
        from ascend import fidelity as fid  # noqa: PLC0415

        print("fidelity suite: 4 conditions x 30 trials ...", flush=True)
        fly_ids = np.array(sorted(idx, key=idx.get), dtype=np.int64)
        fidelity = fid.suite(
            lambda st: runner.make(fid.stimulated(st, path_comp)), runner.n_neurons, runner.params.dt, fly_ids
        )
    criteria = {"phase": 0, "note": "Phase 0 baseline, no pass/fail."}
    if args.engine == "ascend":
        criteria = {
            "phase": 1,
            "prereg": "docs/PREREG_PHASE1.md",
            "S1_rtf_median_ge_1": med >= RTF_TARGET,
            "F_fidelity_suite": None if fidelity is None else fidelity["pass"],
            "consumer_gpu": "reviewed by maintainers from hardware.gpus",
        }
    result = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(["python", *sys.argv]),
        "engine": {"name": args.engine, **runner.info, "dt_ms": 0.1},
        "connectome": {"dataset": dataset, "neurons": runner.n_neurons, **connectome_info(path_con, entries)},
        "workload": {
            "stimulus": args.stimulus,
            "n_stimulated": len(exc),
            "rate_hz": args.rate_hz,
            "t_run_s": args.duration,
            "trials": args.trials,
            "seeds": seeds,
        },
        "timing": {
            "build_s": round(runner.build_s, 3),
            "warmup_s": round(warmup_s, 3),
            "run_s": [round(r, 5) for r in run_s],
            "rtf": [round(x, 5) for x in rtf],
            "rtf_median": round(med, 5),
            "rtf_min": round(min(rtf), 5),
            "rtf_max": round(max(rtf), 5),
            "slowdown_vs_realtime": round(1 / med, 2),
        },
        "fidelity": {
            "spikes_per_trial": [sum(len(v) for v in sp.values()) for sp in spikes],
            "readouts_rate_hz": {
                k: {"mean": round(float(mean[i]), 3), "std": round(float(std[i]), 3)} for k, i in readouts.items()
            },
        },
        "selftest": selftest,
        "fidelity_suite": fidelity,
        "criteria": criteria,
        "valid_submission": not args.synthetic,
        "hardware": hardware.info(),
    }
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}")
    print(f"RTF median {med:.4f}" + (f" ({1 / med:.1f}x slower than real time)" if med < 1 else ""))
    if fidelity is not None:
        print(f"fidelity suite: {'PASS' if fidelity['pass'] else 'FAIL'}")
    if tmp:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
