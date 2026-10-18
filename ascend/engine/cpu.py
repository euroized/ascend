"""Event-driven CPU engine (NumPy). In float64 it computes the reference model exactly.

Every step follows Brian2's schedule for the reference network:

1. groups:     not_refractory = steps since last spike >= refractory steps;
               non-refractory neurons integrate v and g exactly over one dt.
2. thresholds: spikes = (v > v_th) & not_refractory; spiking neurons become refractory.
3. synapses:   spikes from `delay` steps ago add w to g of their non-refractory targets.
               Stimulus input adds w_poi to v of non-refractory stimulated neurons.
4. resets:     spiking neurons get v = v_rst, g = 0.

Synaptic input is summed per step in Brian2's order (spiking neurons ascending, then each
neuron's synapses in file order), so float64 results are bit-identical to Brian2's numpy target.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from . import rng
from .network import Network
from .params import Params


@dataclass
class Spikes:
    """Spike events of one run: parallel arrays of time step and neuron index, in time order."""

    steps: np.ndarray
    neurons: np.ndarray
    dt: float

    def __len__(self) -> int:
        return len(self.steps)

    def trains(self) -> dict[int, np.ndarray]:
        """neuron index -> spike times (s), as the reference wrapper returns them."""
        order = np.lexsort((self.steps, self.neurons))
        n, s = self.neurons[order], self.steps[order]
        cut = np.flatnonzero(np.diff(n)) + 1
        return {int(k[0]): v * self.dt for k, v in zip(np.split(n, cut), np.split(s, cut), strict=True) if len(k)}


class PoissonInput:
    """Bernoulli(rate * dt) kicks per stimulated neuron and step, from the counter-based hash in rng.py.

    Identical on every backend for the same seed.
    """

    def __init__(self, rate_hz: float, seed: int):
        self.rate_hz, self.seed = rate_hz, seed

    def start(self, stim: np.ndarray, dt: float) -> None:
        self._stim, self._thr = stim, rng.threshold(self.rate_hz, dt)

    def kicks(self, step: int) -> np.ndarray:
        """Positions (into the stimulated set) that receive a kick this step."""
        return np.flatnonzero(rng.kicks(self.seed, step, self._stim, self._thr))


class EventInput:
    """Explicit kicks: (step, position in the stimulated set) pairs. Used to replay identical input."""

    def __init__(self, steps: np.ndarray, positions: np.ndarray):
        order = np.argsort(steps, kind="stable")
        self._steps, self._pos = np.asarray(steps)[order], np.asarray(positions)[order]

    def start(self, stim: np.ndarray, dt: float) -> None:
        self._bounds = {}
        if len(self._steps):
            u, first = np.unique(self._steps, return_index=True)
            last = np.append(first[1:], len(self._steps))
            self._bounds = {int(s): (a, b) for s, a, b in zip(u, first, last, strict=True)}

    def kicks(self, step: int) -> np.ndarray:
        a, b = self._bounds.get(step, (0, 0))
        return self._pos[a:b]


class CPUEngine:
    def __init__(
        self,
        net: Network,
        params: Params,
        stimulated: Iterable[int] = (),
        dtype=np.float64,
        shuffle_seed: int | None = None,
    ):
        """shuffle_seed: sum each step's synaptic input in a random order, as GPU atomics may (a control)."""
        self.net, self.p, self.dtype = net, params, np.dtype(dtype)
        self._shuffle = None if shuffle_seed is None else np.random.default_rng(shuffle_seed)
        self.stim = np.asarray(sorted(set(stimulated)), dtype=np.int64)
        if net.n_neurons >= 1 << 20:
            raise ValueError("neuron indices must fit in 20 bits (rng.kicks key layout)")
        self.ref_steps = np.full(net.n_neurons, params.refractory_steps, dtype=np.int64)
        self.ref_steps[self.stim] = 0  # upstream poi(): no refractory period for stimulated neurons
        self.weight = net.weight.astype(self.dtype)
        self.reset()

    def warmup(self, n_steps: int) -> None:
        """Nothing to compile or capture on the CPU; present for a uniform engine interface."""

    def reset(self) -> None:
        n = self.net.n_neurons
        self.v = np.full(n, self.p.v_0, dtype=self.dtype)
        self.g = np.zeros(n, dtype=self.dtype)
        self.last = np.full(n, -(10**12), dtype=np.int64)  # never spiked (Brian2: lastspike = -1e4 s)
        self.step_index = 0
        self._queue: dict[int, np.ndarray] = {}

    def run(self, n_steps: int, stimulus: PoissonInput | EventInput | None = None) -> Spikes:
        p = self.p
        decay_g, offset_v, g_to_v, decay_v = (self.dtype.type(x) for x in p.integration())
        v_th, v_rst, w_poi = (self.dtype.type(x) for x in (p.v_th, p.v_rst, p.w_poi))
        delay = p.delay_steps
        if stimulus is not None:
            stimulus.start(self.stim, p.dt)
        v, g, last, ref, stim = self.v, self.g, self.last, self.ref_steps, self.stim
        out_steps, out_neurons = [], []

        for step in range(self.step_index, self.step_index + n_steps):
            # 1. groups
            nr = (step - last) >= ref
            g_new = np.where(nr, decay_g * g, g)
            v_new = np.where(nr, offset_v + ((g_to_v * g) + (decay_v * v)), v)
            g[:], v[:] = g_new, v_new
            # 2. thresholds
            spk = np.flatnonzero((v > v_th) & nr)
            nr[spk] = False
            last[spk] = step
            # 3. synapses: delayed spikes, then stimulus
            if len(spk):
                self._queue[step + delay] = spk
            arriving = self._queue.pop(step, None)
            if arriving is not None:
                syn = self.net.out_synapses(arriving)
                if self._shuffle is not None:
                    syn = self._shuffle.permutation(syn)
                post = self.net.post[syn]
                keep = nr[post]
                np.add.at(g, post[keep], self.weight[syn][keep])
            if stimulus is not None and len(stim):
                kicked = stim[stimulus.kicks(step)]
                kicked = kicked[nr[kicked]]
                v[kicked] += w_poi
            # 4. resets
            v[spk] = v_rst
            g[spk] = 0
            if len(spk):
                out_steps.append(np.full(len(spk), step, dtype=np.int64))
                out_neurons.append(spk)

        self.step_index += n_steps
        cat = lambda xs: np.concatenate(xs) if xs else np.empty(0, dtype=np.int64)  # noqa: E731
        return Spikes(cat(out_steps), cat(out_neurons), p.dt)
