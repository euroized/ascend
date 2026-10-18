"""Connectivity in compressed sparse row (CSR) form, indexed by presynaptic neuron.

Row order and the order of synapses within a row follow the connectivity file, which is the
synapse order Brian2 uses. The float64 engine relies on this to sum synaptic input in the same
order as the reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Network:
    n_neurons: int
    row_ptr: np.ndarray  # int64, shape (n_neurons + 1,)
    post: np.ndarray  # int32, postsynaptic index of each synapse, grouped by presynaptic neuron
    weight: np.ndarray  # float64 volts, per synapse

    @property
    def n_connections(self) -> int:
        return int(len(self.post))

    @classmethod
    def from_arrays(cls, n_neurons: int, pre: np.ndarray, post: np.ndarray, signed_count: np.ndarray, w_syn: float):
        order = np.argsort(pre, kind="stable")
        counts = np.bincount(pre, minlength=n_neurons)
        row_ptr = np.zeros(n_neurons + 1, dtype=np.int64)
        np.cumsum(counts, out=row_ptr[1:])
        # Same arithmetic as the reference: int64 array * float.
        weight = np.asarray(signed_count, dtype=np.int64)[order] * w_syn
        return cls(n_neurons, row_ptr, np.asarray(post, dtype=np.int32)[order], weight)

    @classmethod
    def from_files(cls, path_comp: Path, path_con: Path, w_syn: float) -> Network:
        n = len(pd.read_csv(path_comp, index_col=0))
        cols = ["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"]
        con = pd.read_parquet(path_con, columns=cols)
        return cls.from_arrays(n, *(con[c].to_numpy() for c in cols), w_syn=w_syn)

    def out_synapses(self, neurons: np.ndarray) -> np.ndarray:
        """Synapse indices of all outgoing synapses of `neurons`, neuron by neuron, in row order."""
        if len(neurons) == 0:
            return np.empty(0, dtype=np.int64)
        starts, ends = self.row_ptr[neurons], self.row_ptr[neurons + 1]
        lengths = ends - starts
        offsets = np.repeat(starts - np.concatenate(([0], np.cumsum(lengths)[:-1])), lengths)
        return np.arange(lengths.sum(), dtype=np.int64) + offsets
