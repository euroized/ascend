"""Small random connectomes in the reference file format, for tests and CI smoke runs.

Results on a synthetic connectome are never valid benchmark results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FIRST_ID = 720575940600000000


def write(
    directory: Path,
    n_neurons: int = 300,
    n_connections: int = 6000,
    inhibitory_frac: float = 0.2,
    max_synapses: int = 30,
    seed: int = 0,
) -> tuple[Path, Path]:
    """Write Completeness/Connectivity files with the same columns as the FlyWire 783 files."""
    rng = np.random.default_rng(seed)
    ids = FIRST_ID + np.arange(n_neurons, dtype=np.int64)
    pairs = set()
    while len(pairs) < n_connections:
        i, j = rng.integers(0, n_neurons, size=2)
        if i != j:
            pairs.add((int(i), int(j)))
    pre, post = np.array(sorted(pairs)).T
    sign = np.where(rng.random(n_neurons) < inhibitory_frac, -1, 1)
    syn = rng.integers(1, max_synapses + 1, size=len(pre))
    con = pd.DataFrame(
        {
            "Presynaptic_ID": ids[pre],
            "Postsynaptic_ID": ids[post],
            "Presynaptic_Index": pre,
            "Postsynaptic_Index": post,
            "Connectivity": syn,
            "Excitatory": sign[pre],
            "Excitatory x Connectivity": syn * sign[pre],
        }
    ).astype("int64")
    comp = pd.DataFrame({"Completed": True}, index=pd.Index(ids))
    directory.mkdir(parents=True, exist_ok=True)
    path_comp, path_con = directory / "Completeness_synthetic.csv", directory / "Connectivity_synthetic.parquet"
    comp.to_csv(path_comp)
    con.to_parquet(path_con)
    return path_comp, path_con
