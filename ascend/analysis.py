"""Firing rates from spike trains, with the same definitions as Shiu et al.'s utils.get_rate()."""

from __future__ import annotations

import numpy as np


def counts(trials: list[dict[int, np.ndarray]], n_neurons: int) -> np.ndarray:
    """Spike counts, shape (n_trials, n_neurons)."""
    c = np.zeros((len(trials), n_neurons), dtype=np.int64)
    for t, spikes in enumerate(trials):
        for i, s in spikes.items():
            c[t, i] = len(s)
    return c


def rates(trials: list[dict[int, np.ndarray]], n_neurons: int, t_run_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation (ddof=0, as upstream) of each neuron's rate in Hz across trials.

    Trials in which a neuron is silent count as 0 Hz.
    """
    r = counts(trials, n_neurons) / t_run_s
    return r.mean(axis=0), r.std(axis=0)
