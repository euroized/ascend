"""Quick consistency check of a device engine against the CPU engine, on a small synthetic network.

Both engines run in float32 with the same stimulus kicks. The CPU engine sums synaptic input in a
fixed order and the GPU in atomic order, so spike trains can differ in rare near-threshold cases;
they should agree almost entirely. This is a diagnostic recorded with benchmark results, not a
preregistered criterion.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from .. import synthetic
from . import CPUEngine, Network, Params, PoissonInput, make


def run(device: str, n_steps: int = 2000) -> dict:
    with tempfile.TemporaryDirectory() as d:
        pc, pn = synthetic.write(Path(d), n_neurons=500, n_connections=40000, seed=3)
        p = Params.reference()
        net = Network.from_files(pc, pn, p.w_syn)
    stim = list(range(0, 500, 20))
    ref = CPUEngine(net, p, stim, dtype=np.float32).run(n_steps, PoissonInput(150, 11))
    dev = make(net, p, stim, device=device, dtype=np.float32)
    got = dev.run(n_steps, PoissonInput(150, 11))
    a = set(zip(ref.steps.tolist(), ref.neurons.tolist(), strict=True))
    b = set(zip(got.steps.tolist(), got.neurons.tolist(), strict=True))
    agree = len(a & b) / max(len(a | b), 1)
    return {
        "device": device,
        "n_steps": n_steps,
        "spikes_cpu": len(ref),
        "spikes_device": len(got),
        "identical": a == b,
        "jaccard": round(agree, 6),
        "ok": agree >= 0.99,
    }
