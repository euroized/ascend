"""Same-input tier (G): two engines given identical stimulus kicks must emit the same spikes.

Added by Amendment 1 to docs/PREREG_PHASE1.md. An engine under test runs each workload once,
with the counter-based hash kicks of seed KICK_SEED, and is compared with the CPU float32 engine
as built. Agreement is the Jaccard index of the two sets of (step, neuron) spike events.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from .engine import CPUEngine, Network, Params, PoissonInput, Spikes

KICK_SEED = 1
RATE_HZ = 150.0
T_RUN_S = 1.0
WORKLOADS = ("sugar", "bitter", "jon_all")
MIN_JACCARD = 0.99


def jaccard(a: Spikes, b: Spikes) -> float:
    x = set(zip(a.steps.tolist(), a.neurons.tolist(), strict=True))
    y = set(zip(b.steps.tolist(), b.neurons.tolist(), strict=True))
    return len(x & y) / len(x | y) if x | y else 1.0


def compare(make_engine: Callable, net: Network, params: Params, stimulated: Callable, log=print) -> dict:
    """make_engine(stimulated_indices) -> engine under test. stimulated(workload) -> indices."""
    n_steps = int(round(T_RUN_S / params.dt))
    out = {}
    for w in WORKLOADS:
        stim = stimulated(w)
        t0 = time.time()
        ref = CPUEngine(net, params, stim, dtype=np.float32).run(n_steps, PoissonInput(RATE_HZ, KICK_SEED))
        eng = make_engine(stim)
        if hasattr(eng, "warmup"):
            eng.warmup(n_steps)
        got = eng.run(n_steps, PoissonInput(RATE_HZ, KICK_SEED))
        j = jaccard(ref, got)
        out[w] = {
            "spikes_reference": len(ref),
            "spikes_engine": len(got),
            "jaccard": round(j, 6),
            "pass": bool(j >= MIN_JACCARD),
        }
        log(f"{w}: jaccard {j:.4f} ({len(got)} vs {len(ref)} spikes, {time.time() - t0:.0f}s)")
    return {
        "prereg": "docs/PREREG_PHASE1.md#amendment-1",
        "kick_seed": KICK_SEED,
        "min_jaccard": MIN_JACCARD,
        "pass": all(r["pass"] for r in out.values()),
        "workloads": out,
    }
