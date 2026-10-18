"""Host side of the kernel engine, shared by the CUDA backend and the CPU emulator of its kernels.

A run is split into chunks of K steps. Each chunk launches, per step, the kernels `update`,
`deliver` and `clear` from kernels.cu, then `advance` once. On CUDA the chunk is captured as one
CUDA Graph and replayed, so the host is involved once per chunk, not once per step. Spikes are
recorded on the device and read back after each chunk.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from . import rng
from .cpu import PoissonInput, Spikes
from .network import Network
from .params import Params

KERNELS = Path(__file__).with_name("kernels.cu")
NEVER = -(10**12)


class KernelEngine:
    """Subclasses provide `_array` (host -> backend), `_numpy` (backend -> host), `_zero` and `_launch`."""

    chunk_steps = 1000
    update_threads: int
    deliver_threads: int

    def __init__(
        self, net: Network, params: Params, stimulated: Iterable[int] = (), dtype=np.float32, rec_cap: int = 1 << 22
    ):
        if net.n_neurons >= 1 << 20:
            raise ValueError("neuron indices must fit in 20 bits (rng.kicks key layout)")
        self.net, self.p, self.real = net, params, np.dtype(dtype)
        self.dtype = self.real
        n, d = net.n_neurons, params.delay_steps
        self.n, self.delay = n, d
        stim = np.asarray(sorted(set(stimulated)), dtype=np.int64)
        ref = np.full(n, params.refractory_steps, dtype=np.int32)
        ref[stim] = 0
        is_stim = np.zeros(n, dtype=np.uint8)
        is_stim[stim] = 1
        self.rec_cap = int(rec_cap)

        a = self._array
        self.ref, self.is_stim = a(ref), a(is_stim)
        self.row_ptr, self.post, self.w = a(net.row_ptr), a(net.post), a(net.weight.astype(self.real))
        self.v, self.g = a(np.zeros(n, self.real)), a(np.zeros(n, self.real))
        self.last, self.nr = a(np.zeros(n, np.int64)), a(np.zeros(n, np.uint8))
        self.ring_buf, self.ring_cnt = a(np.zeros((d + 1) * n, np.int32)), a(np.zeros(d + 1, np.int32))
        self.rec_step, self.rec_neuron = a(np.zeros(self.rec_cap, np.int64)), a(np.zeros(self.rec_cap, np.int32))
        self.rec_cnt, self.base_step = a(np.zeros(1, np.uint32)), a(np.zeros(1, np.int64))
        self.rng = a(np.zeros(2, np.uint64))
        r = self.real.type
        decay_g, offset_v, g_to_v, decay_v = params.integration()
        self.consts = tuple(r(x) for x in (decay_g, offset_v, g_to_v, decay_v, params.v_th, params.v_rst, params.w_poi))
        self.reset()

    def reset(self) -> None:
        self._set(self.v, np.full(self.n, self.p.v_0, self.real))
        self._set(self.g, np.zeros(self.n, self.real))
        self._set(self.last, np.full(self.n, NEVER, np.int64))
        self._set(self.ring_cnt, np.zeros(self.delay + 1, np.int32))
        self._set(self.base_step, np.zeros(1, np.int64))

    # --- one chunk of kernel launches -----------------------------------------------------------
    def _launch_chunk(self, steps: int) -> None:
        n, d = np.int32(self.n), np.int32(self.delay)
        self._zero(self.rec_cnt)
        for k in range(steps):
            k = np.int32(k)
            self._launch(
                "update",
                self.update_threads,
                (
                    n,
                    self.base_step,
                    k,
                    self.v,
                    self.g,
                    self.last,
                    self.ref,
                    self.nr,
                    self.is_stim,
                    self.rng,
                    *self.consts,
                    d,
                    self.ring_buf,
                    self.ring_cnt,
                    self.rec_step,
                    self.rec_neuron,
                    self.rec_cnt,
                    np.uint32(self.rec_cap),
                ),
            )
            self._launch(
                "deliver",
                self.deliver_threads,
                (
                    n,
                    self.base_step,
                    k,
                    d,
                    self.ring_buf,
                    self.ring_cnt,
                    self.row_ptr,
                    self.post,
                    self.w,
                    self.nr,
                    self.g,
                ),
            )
            self._launch("clear", 32, (self.base_step, k, d, self.ring_cnt))
        self._launch("advance", 32, (self.base_step, np.int32(steps)))

    def _run_chunk(self, steps: int) -> None:
        self._launch_chunk(steps)

    def run(self, n_steps: int, stimulus: PoissonInput | None = None) -> Spikes:
        seed, thr = (stimulus.seed, rng.threshold(stimulus.rate_hz, self.p.dt)) if stimulus else (0, 0)
        self._set(self.rng, np.array([seed, thr], dtype=np.uint64))
        start = int(self._numpy(self.base_step)[0])
        steps, neurons, done = [], [], 0
        while done < n_steps:
            k = min(self.chunk_steps, n_steps - done)
            self._run_chunk(k)
            cnt = int(self._numpy(self.rec_cnt)[0])
            if cnt > self.rec_cap:
                raise RuntimeError(f"spike record overflow: {cnt} spikes in {k} steps, capacity {self.rec_cap}")
            steps.append(self._numpy(self.rec_step[:cnt]))
            neurons.append(self._numpy(self.rec_neuron[:cnt]).astype(np.int64))
            done += k
        s, nn = np.concatenate(steps), np.concatenate(neurons)
        order = np.lexsort((nn, s))  # device append order is arbitrary within a step
        assert start + n_steps == int(self._numpy(self.base_step)[0])
        return Spikes(s[order], nn[order], self.p.dt)

    def warmup(self, n_steps: int) -> None:
        """Run once untimed so one-time costs (compilation, graph capture) stay out of timings, then reset."""
        self._run_chunk(min(n_steps, self.chunk_steps))
        self.reset()

    def state(self) -> tuple[np.ndarray, np.ndarray]:
        return self._numpy(self.v), self._numpy(self.g)

    # --- backend hooks --------------------------------------------------------------------------
    def _array(self, x: np.ndarray):
        raise NotImplementedError

    def _numpy(self, x) -> np.ndarray:
        raise NotImplementedError

    def _set(self, dst, src: np.ndarray) -> None:
        raise NotImplementedError

    def _zero(self, x) -> None:
        raise NotImplementedError

    def _launch(self, kernel: str, threads: int, args: tuple) -> None:
        raise NotImplementedError
