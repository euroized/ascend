"""The ASCEND engine: event-driven simulation of the reference model (Phase 1).

Backends: "cpu" (NumPy; float64 matches Brian2 spike for spike), "cuda" (CuPy + CUDA Graphs),
and "emulate" (kernels.cu run on the CPU, for tests).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .cpu import CPUEngine, EventInput, PoissonInput, Spikes
from .network import Network
from .params import Params

__all__ = ["CPUEngine", "EventInput", "Network", "Params", "PoissonInput", "Spikes", "make"]


def make(net: Network, params: Params, stimulated: Iterable[int] = (), device: str = "cpu", dtype=None):
    """Engine for `device`. Default precision: float64 on "cpu", float32 elsewhere."""
    if device == "cpu":
        return CPUEngine(net, params, stimulated, dtype=dtype or np.float64)
    if device == "cuda":
        from .cuda import CUDAEngine  # noqa: PLC0415

        return CUDAEngine(net, params, stimulated, dtype=dtype or np.float32)
    if device == "emulate":
        from .emulate import EmulatedEngine  # noqa: PLC0415

        return EmulatedEngine(net, params, stimulated, dtype=dtype or np.float32)
    raise ValueError(f"unknown device {device!r}")
