"""Counter-based random kicks, identical on every backend.

Each (seed, step, neuron) triple is hashed with the SplitMix64 finaliser. A kick happens when
the top 32 bits fall below floor(p * 2**32), where p = rate * dt. The CUDA kernels
(kernels.cu, `kick()`) compute exactly the same integers, so CPU and GPU engines given the
same seed receive the same stimulus.
"""

from __future__ import annotations

import numpy as np

GOLDEN = np.uint64(0x9E3779B97F4A7C15)
M1 = np.uint64(0xBF58476D1CE4E5B9)
M2 = np.uint64(0x94D049BB133111EB)


def _mix(x: np.ndarray) -> np.ndarray:
    x = x ^ (x >> np.uint64(30))
    x = x * M1
    x = x ^ (x >> np.uint64(27))
    x = x * M2
    return x ^ (x >> np.uint64(31))


def threshold(rate_hz: float, dt: float) -> int:
    """32-bit threshold for a kick probability of rate_hz * dt per step."""
    return min(int(np.floor(rate_hz * dt * 2.0**32)), 2**32)


def kicks(seed: int, step: int, neurons: np.ndarray, thr: int) -> np.ndarray:
    """Boolean mask: which of `neurons` (global indices) get a kick at `step`."""
    with np.errstate(over="ignore"):
        key = np.uint64(seed + 1) * GOLDEN + (np.uint64(step) << np.uint64(20)) + neurons.astype(np.uint64)
        return (_mix(key) >> np.uint64(32)) < np.uint64(thr)
