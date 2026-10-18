"""Record the runs shown by the demo viewer (docs/demo/) and write docs/demo/data.js.

    python scripts/fetch_data.py --group provenance   # FlyWire annotations, for neuron positions
    python scripts/make_demo.py

Every run is the full FlyWire v783 model on the CPU float64 engine (identical to the Brian2
reference for the same input), 1.0 s, Poisson input at 150 Hz with kick seed 0:
real wiring with sugar, bitter and Johnston's organ input, and the `rewired` null network
(shuffle seed 0, docs/PREREG_BASELINES.md) with sugar input. Output is deterministic.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ascend import baselines, data, fidelity, stimuli  # noqa: E402
from ascend.engine import CPUEngine, Params, PoissonInput  # noqa: E402

OUT = data.REPO_ROOT / "docs" / "demo" / "data.js"
KICK_SEED, RATE_HZ, T_RUN_S, CLOUD_N = 0, 150.0, 1.0, 20_000
RUNS = [  # key, wiring, stimulus, label, description
    ("sugar", "real", "sugar", "Sugar", "21 sugar-sensing gustatory neurons driven at 150 Hz."),
    ("bitter", "real", "bitter", "Bitter", "21 bitter-sensing gustatory neurons driven at 150 Hz."),
    ("jon", "real", "jon_all", "Johnston's organ", "146 antennal mechanosensory neurons (JONs) driven at 150 Hz."),
    (
        "scrambled",
        "rewired",
        "sugar",
        "Sugar, scrambled wiring",
        "The same sugar input into a null network: every connection's target shuffled, "
        "each neuron's degree and synapse weights kept.",
    ),
]
VOXEL_NM = np.array([4.0, 4.0, 40.0])  # FlyWire annotation coordinates are in 4 x 4 x 40 nm voxels


def positions(path_comp: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Engine index -> (x, y, z) in µm (NaN where unannotated), and the annotations by engine index."""
    idx = stimuli.id_to_index(path_comp)
    ann = pd.read_csv(
        data.path("flywire_neuron_annotations.tsv"),
        sep="\t",
        usecols=["root_id", "pos_x", "pos_y", "pos_z", "cell_type", "super_class"],
        low_memory=False,
    )
    ann = ann[ann.root_id.isin(idx)].copy()
    ann["i"] = ann.root_id.map(idx)
    ann = ann.set_index("i").sort_index()
    xyz = np.full((len(idx), 3), np.nan)
    xyz[ann.index] = ann[["pos_x", "pos_y", "pos_z"]].to_numpy() * VOXEL_NM / 1000
    return xyz, ann


def main() -> int:
    p = Params.reference()
    path_comp, path_con = data.path("Completeness_783.csv"), data.path("Connectivity_783.parquet")
    n = len(pd.read_csv(path_comp, index_col=0))
    cols = ["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"]
    con = pd.read_parquet(path_con, columns=cols)
    arrays = [con[c].to_numpy() for c in cols]
    del con
    xyz, ann = positions(path_comp)
    origin = np.nanmin(xyz, axis=0)
    q = lambda a: np.round((a - origin) * 2).astype(int)  # noqa: E731  (0.5 µm grid)
    readouts = stimuli.readouts()
    idx = stimuli.id_to_index(path_comp)
    n_steps = int(round(T_RUN_S / p.dt))

    rng = np.random.default_rng(0)
    have = np.flatnonzero(~np.isnan(xyz[:, 0]))
    cloud = q(xyz[np.sort(rng.choice(have, CLOUD_N, replace=False))])

    nets, runs = {}, {}
    for key, wiring, stim, label, desc in RUNS:
        if wiring not in nets:
            nets[wiring] = baselines.network(wiring, 0, n, *arrays, w_syn=p.w_syn)
        stimulated = fidelity.stimulated(stim, path_comp)
        s = CPUEngine(nets[wiring], p, stimulated, dtype=np.float64).run(n_steps, PoissonInput(RATE_HZ, KICK_SEED))
        active, local = np.unique(s.neurons, return_inverse=True)
        counts = np.bincount(local, minlength=len(active))
        pos = xyz[active]
        located = ~np.isnan(pos[:, 0])
        stim_set = set(stimulated)
        downstream = np.array([a not in stim_set for a in active.tolist()], dtype=bool)
        top = [i for i in np.argsort(-counts, kind="stable") if downstream[i]][:8]
        runs[key] = {
            "label": label,
            "description": desc,
            "wiring": wiring,
            "spikes": int(len(s)),
            "active": int(len(active)),
            # neurons that spiked: x, y, z on a 0.5 µm grid (-1 if unannotated), stimulated flag
            "neurons": [
                [*(q(pos[i]).tolist() if located[i] else [-1, -1, -1]), int(active[i] in stim_set)]
                for i in range(len(active))
            ],
            "steps": np.diff(s.steps, prepend=0).tolist(),  # delta-encoded spike steps (0.1 ms)
            "who": local.tolist(),  # index into neurons
            "readouts": {
                k: (int(np.searchsorted(active, idx[f])) if idx[f] in set(active.tolist()) else -1)
                for k, f in readouts.items()
            },
            "top": [  # most active neurons that were not stimulated directly
                {
                    "type": str(ann.cell_type.get(active[i], "") or "unnamed"),
                    "class": str(ann.super_class.get(active[i], "") or ""),
                    "spikes": int(counts[i]),
                }
                for i in top
            ],
        }
        print(f"{key}: {len(s)} spikes, {len(active)} active neurons", flush=True)

    payload = {
        "dt_ms": p.dt * 1000,
        "steps": n_steps,
        "kick_seed": KICK_SEED,
        "rate_hz": RATE_HZ,
        "extent": q(np.nanmax(xyz, axis=0)).tolist(),
        "cloud": cloud.ravel().tolist(),
        "runs": runs,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = "window.ASCEND_DEMO = " + json.dumps(payload, separators=(",", ":")) + ";\n"
    OUT.write_text(text)
    print(f"wrote {OUT} ({len(text) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
