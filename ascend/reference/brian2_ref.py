"""Seeded, timed trials of the Shiu et al. 2024 reference model.

The model itself is shiu_model.py, a verbatim copy of model.py from
philshiu/Drosophila_brain_model @ 91bdd1e (MIT, see LICENSE.shiu). This module only calls its
functions: it builds the network exactly as upstream run_trial() does, then seeds Brian2's
random number generators and times the build and the run separately.

Upstream rebuilds the network for every trial. Session builds it once, stores its initial
state, and restores that state before each trial. With the same seed, both produce identical
spike trains (tests/test_reference.py checks this), and Session skips the rebuild.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import brian2 as b2
import numpy as np

from . import shiu_model as shiu


@dataclass
class Trial:
    seed: int
    t_run_s: float
    run_s: float
    spikes: dict[int, np.ndarray] = field(repr=False)  # neuron index -> spike times (s)

    @property
    def rtf(self) -> float:
        """Real-time factor: simulated seconds per wall-clock second of net.run()."""
        return self.t_run_s / self.run_s

    @property
    def n_spikes(self) -> int:
        return int(sum(len(v) for v in self.spikes.values()))


def params(t_run_s: float = 1.0, r_poi_hz: float | None = None, r_poi2_hz: float | None = None) -> dict:
    """Upstream default_params with optional overrides of trial length and stimulus rates."""
    p = dict(shiu.default_params)
    p["t_run"] = t_run_s * b2.second
    if r_poi_hz is not None:
        p["r_poi"] = r_poi_hz * b2.Hz
    if r_poi2_hz is not None:
        p["r_poi2"] = r_poi2_hz * b2.Hz
    return p


def _spikes(spk_mon) -> dict[int, np.ndarray]:
    return {int(k): np.asarray(v / b2.second, dtype=np.float64) for k, v in shiu.get_spk_trn(spk_mon).items()}


class Session:
    """A network built once for one condition (stimulus, rates, silencing) and run trial by trial.

    exc, exc2 and slnc are neuron indices (rows of the completeness file).
    """

    def __init__(
        self,
        path_comp: Path,
        path_con: Path,
        exc: list[int],
        *,
        exc2: list[int] = (),
        slnc: list[int] = (),
        t_run_s: float = 1.0,
        r_poi_hz: float | None = None,
        r_poi2_hz: float | None = None,
        target: str = "cython",
    ):
        b2.prefs.codegen.target = target
        self.params = params(t_run_s, r_poi_hz, r_poi2_hz)
        self.t_run_s = t_run_s

        t0 = time.perf_counter()
        neu, syn, self.spk_mon = shiu.create_model(str(path_comp), str(path_con), self.params)
        poi_inp, neu = shiu.poi(neu, list(exc), list(exc2), self.params)
        syn = shiu.silence(list(slnc), syn)
        self.net = b2.Network(neu, syn, self.spk_mon, *poi_inp)
        self.n_neurons = len(neu)
        self.net.store("init")
        self.build_s = time.perf_counter() - t0
        self.warmup_s: float | None = None

    def warmup(self) -> float:
        """Run one time step and restore, so code generation and spike-queue setup happen untimed."""
        t0 = time.perf_counter()
        self.net.run(b2.defaultclock.dt)
        self.net.restore("init")
        self.warmup_s = time.perf_counter() - t0
        return self.warmup_s

    def trial(self, seed: int) -> Trial:
        self.net.restore("init")
        b2.seed(seed)
        t0 = time.perf_counter()
        self.net.run(duration=self.params["t_run"])
        run_s = time.perf_counter() - t0
        return Trial(seed=seed, t_run_s=self.t_run_s, run_s=run_s, spikes=_spikes(self.spk_mon))


def run_trial(
    path_comp: Path,
    path_con: Path,
    exc: list[int],
    *,
    seed: int,
    exc2: list[int] = (),
    slnc: list[int] = (),
    t_run_s: float = 1.0,
    r_poi_hz: float | None = None,
    r_poi2_hz: float | None = None,
    target: str = "cython",
) -> Trial:
    """One trial on a freshly built network, as upstream run_trial() does, plus seeding."""
    b2.prefs.codegen.target = target
    p = params(t_run_s, r_poi_hz, r_poi2_hz)
    neu, syn, spk_mon = shiu.create_model(str(path_comp), str(path_con), p)
    poi_inp, neu = shiu.poi(neu, list(exc), list(exc2), p)
    syn = shiu.silence(list(slnc), syn)
    net = b2.Network(neu, syn, spk_mon, *poi_inp)
    b2.seed(seed)
    t0 = time.perf_counter()
    net.run(duration=p["t_run"])
    run_s = time.perf_counter() - t0
    return Trial(seed=seed, t_run_s=t_run_s, run_s=run_s, spikes=_spikes(spk_mon))
