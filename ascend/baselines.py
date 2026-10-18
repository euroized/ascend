"""Matched null networks and the readout-selectivity score (docs/PREREG_BASELINES.md).

Each null network has the same neurons, the same number of connections and the same multiset of
signed synaptic weights as the FlyWire v783 network. They differ in which wiring statistics they keep:

    real      the connectome as published
    weights   connections kept, signed weights permuted across all connections
    rewired   each connection keeps its presynaptic neuron and weight, targets permuted across all
              connections (keeps every neuron's in- and out-degree and outgoing weights, so Dale's law)
    random    presynaptic and postsynaptic neuron drawn uniformly at random, weights permuted

Stimulus and readout neurons keep their identities in every variant.
"""

from __future__ import annotations

import numpy as np

from .engine import Network

VARIANTS = ("real", "weights", "rewired", "random")
SHUFFLE_SEEDS = (0, 1, 2)
CONDITIONS = {"sugar": ("sugar", 150.0), "bitter": ("bitter", 150.0), "jon_all": ("jon_all", 150.0)}
SEEDS = list(range(10))
T_RUN_S = 1.0
ON_HZ = 10.0  # a readout group is "on" if its mean rate is >= ON_HZ
OFF_HZ = 1.0  # and "off" if it is < OFF_HZ; anything between counts as wrong either way
RUNAWAY_SPIKES = 1_000_000  # a trial over this many spikes is stopped and the condition marked runaway
CHUNK_STEPS = 100
GROUPS = {"MN9": ("mn9_left", "mn9_right"), "JON_DN": ("jon_dn1", "jon_dn2")}
EXPECTED = {  # condition -> group -> expected state
    "sugar": {"MN9": "on", "JON_DN": "off"},
    "bitter": {"MN9": "off", "JON_DN": "off"},
    "jon_all": {"MN9": "off", "JON_DN": "on"},
}


def null_arrays(variant: str, pre: np.ndarray, post: np.ndarray, signed: np.ndarray, n_neurons: int, seed: int):
    """(pre, post, signed_count) arrays of one variant. `real` returns the inputs unchanged."""
    if variant == "real":
        return pre, post, signed
    rng = np.random.default_rng([seed, VARIANTS.index(variant)])
    if variant == "weights":
        return pre, post, rng.permutation(signed)
    if variant == "rewired":
        return pre, rng.permutation(post), signed
    if variant == "random":
        m = len(pre)
        return rng.integers(0, n_neurons, m), rng.integers(0, n_neurons, m), rng.permutation(signed)
    raise ValueError(f"unknown variant {variant!r}")


def network(variant: str, seed: int, n_neurons: int, pre, post, signed, w_syn: float) -> Network:
    return Network.from_arrays(n_neurons, *null_arrays(variant, pre, post, signed, n_neurons, seed), w_syn=w_syn)


def state(rate_hz: float) -> str:
    return "on" if rate_hz >= ON_HZ else "off" if rate_hz < OFF_HZ else "between"


def score(group_rates: dict[str, dict[str, float]], runaway: dict[str, bool]) -> dict:
    """group_rates[condition][group] = mean rate (Hz). One point per (condition, group) cell matching EXPECTED."""
    cells = []
    for cond, exp in EXPECTED.items():
        for grp, want in exp.items():
            got = "runaway" if runaway.get(cond) else state(group_rates[cond][grp])
            cells.append({"condition": cond, "group": grp, "expected": want, "observed": got, "pass": got == want})
    return {"score": sum(c["pass"] for c in cells), "of": len(cells), "cells": cells}


def run_trial(eng, rate_hz: float, seed: int, n_steps: int) -> tuple[np.ndarray, bool]:
    """Spike count per neuron of one trial, run in chunks; stops early once RUNAWAY_SPIKES is exceeded."""
    from .engine import PoissonInput  # noqa: PLC0415

    eng.reset()
    counts = np.zeros(eng.net.n_neurons, dtype=np.int64)
    done = 0
    while done < n_steps:
        k = min(CHUNK_STEPS, n_steps - done)
        s = eng.run(k, PoissonInput(rate_hz, seed))
        counts += np.bincount(s.neurons, minlength=len(counts))
        done += k
        if counts.sum() > RUNAWAY_SPIKES:
            return counts, True
    return counts, False
