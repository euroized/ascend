"""Run kernels.cu on the CPU: every thread of every kernel in turn, compiled with a C++ compiler.

This tests the exact kernel logic that runs on GPUs, in CI and on machines without one. Atomics
become plain read-modify-writes, which is valid because threads run one after another. It is
slow and meant only for small networks.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .kernel_engine import KERNELS, KernelEngine

WRAPPERS = r"""
#include "kernels.cu"
extern "C" {
void emu_update(i64 nthreads, int n, const i64* base_step, int k, real* v, real* g, i64* last, const int* ref,
                unsigned char* nr_out, const unsigned char* is_stim, const u64* rng, real decay_g, real offset_v,
                real g_to_v, real decay_v, real v_th, real v_rst, real w_poi, int delay, int* ring_buf,
                int* ring_cnt, i64* rec_step, int* rec_neuron, unsigned int* rec_cnt, unsigned int rec_cap) {
    for (i64 t = 0; t < nthreads; ++t)
        update_body(t, nthreads, n, base_step, k, v, g, last, ref, nr_out, is_stim, rng, decay_g, offset_v,
                    g_to_v, decay_v, v_th, v_rst, w_poi, delay, ring_buf, ring_cnt, rec_step, rec_neuron,
                    rec_cnt, rec_cap);
}
void emu_deliver(i64 nthreads, int n, const i64* base_step, int k, int delay, const int* ring_buf,
                 const int* ring_cnt, const i64* row_ptr, const int* post, const real* w,
                 const unsigned char* nr, real* g) {
    for (i64 t = 0; t < nthreads; ++t)
        deliver_body(t, nthreads, n, base_step, k, delay, ring_buf, ring_cnt, row_ptr, post, w, nr, g);
}
void emu_clear(i64 nthreads, const i64* base_step, int k, int delay, int* ring_cnt) {
    clear_body(base_step, k, delay, ring_cnt);
}
void emu_advance(i64 nthreads, i64* base_step, int steps) { *base_step += steps; }
}
"""

_LIBS: dict[str, ctypes.CDLL] = {}


def compiler() -> str | None:
    return os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")


def _library(real: str) -> ctypes.CDLL:
    """Compile (once per source version and REAL type) and load the emulation library."""
    src = KERNELS.read_text() + WRAPPERS + real
    key = hashlib.sha256(src.encode()).hexdigest()[:16]
    if key in _LIBS:
        return _LIBS[key]
    cache = Path(os.environ.get("ASCEND_CACHE", Path(tempfile.gettempdir()) / "ascend-emulate"))
    cache.mkdir(parents=True, exist_ok=True)
    so = cache / f"kernels_{real}_{key}.so"
    if not so.exists():
        cxx = compiler()
        if cxx is None:
            raise RuntimeError("no C++ compiler found (set CXX)")
        wrapper = cache / f"emu_{key}.cpp"
        wrapper.write_text(WRAPPERS)
        tmp = so.with_suffix(".tmp.so")
        subprocess.run(
            [
                cxx,
                "-O2",
                "-shared",
                "-fPIC",
                "-ffp-contract=off",
                "-DASCEND_EMULATE",
                f"-DREAL={real}",
                f"-I{KERNELS.parent}",
                "-o",
                str(tmp),
                str(wrapper),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        tmp.replace(so)
    lib = ctypes.CDLL(str(so))
    _LIBS[key] = lib
    return lib


class EmulatedEngine(KernelEngine):
    """KernelEngine whose kernels run on the CPU. update: one thread per neuron; deliver: 8 warps."""

    deliver_threads = 8 * 32

    def __init__(self, *args, **kwargs):
        dtype = np.dtype(kwargs.get("dtype", np.float32))
        self._lib = _library({np.float32: "float", np.float64: "double"}[dtype.type])
        self._ctype = {np.float32: ctypes.c_float, np.float64: ctypes.c_double}[dtype.type]
        super().__init__(*args, **kwargs)
        self.update_threads = self.n

    def _array(self, x: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(x).copy()

    def _numpy(self, x: np.ndarray) -> np.ndarray:
        return np.array(x)

    def _set(self, dst: np.ndarray, src: np.ndarray) -> None:
        dst[...] = src

    def _zero(self, x: np.ndarray) -> None:
        x[...] = 0

    def _arg(self, a):
        if isinstance(a, np.ndarray):
            return ctypes.c_void_p(a.ctypes.data)
        if isinstance(a, np.floating):
            return self._ctype(a)
        return {np.int32: ctypes.c_int32, np.int64: ctypes.c_int64, np.uint32: ctypes.c_uint32}[type(a)](a)

    def _launch(self, kernel: str, threads: int, args: tuple) -> None:
        getattr(self._lib, f"emu_{kernel}")(ctypes.c_int64(threads), *(self._arg(a) for a in args))
