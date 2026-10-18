"""Same-input tier G (docs/PREREG_PHASE1.md, Amendment 1): an engine variant vs the CPU float32 engine.

Variants:
    shuffle3, shuffle4   CPU float32, synaptic input summed in random order (G-pos: must pass)
    delay17              CPU float32, synaptic delay 17 steps          (G-neg-delay: must fail)
    weight95             CPU float32, weights x 0.95                   (G-neg-weight: must fail)
    device               the engine on --device (cuda, emulate)        (GPU submissions)

    python scripts/sameinput_check.py --variant shuffle3 --out results/phase1/sameinput_shuffle3.json
"""

import argparse
import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ascend import data, fidelity, hardware, sameinput  # noqa: E402
from ascend.engine import CPUEngine, Network, Params, make  # noqa: E402

VARIANTS = ("shuffle3", "shuffle4", "delay17", "weight95", "device")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--device", default="cuda", choices=("cuda", "emulate"), help="for --variant device")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    p = Params.reference()
    path_comp, path_con = data.path("Completeness_783.csv"), data.path("Connectivity_783.parquet")
    net = Network.from_files(path_comp, path_con, p.w_syn)
    f32 = np.float32
    variant = {
        "shuffle3": lambda st: CPUEngine(net, p, st, dtype=f32, shuffle_seed=3),
        "shuffle4": lambda st: CPUEngine(net, p, st, dtype=f32, shuffle_seed=4),
        "delay17": lambda st: CPUEngine(net, dataclasses.replace(p, t_dly=17 * p.dt), st, dtype=f32),
        "weight95": lambda st: CPUEngine(dataclasses.replace(net, weight=net.weight * 0.95), p, st, dtype=f32),
        "device": lambda st: make(net, p, st, device=args.device, dtype=f32),
    }[args.variant]

    res = sameinput.compare(variant, net, p, lambda w: fidelity.stimulated(w, path_comp))
    result = {
        "schema": "ascend-sameinput/1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "variant": args.variant,
        "device": args.device if args.variant == "device" else "cpu",
        **res,
        "hardware": hardware.info(),
    }
    print(f"{args.variant}: Jaccard >= {sameinput.MIN_JACCARD} in all workloads: {res['pass']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
