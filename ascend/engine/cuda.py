"""CUDA backend: kernels.cu compiled by NVRTC through CuPy, each chunk captured once as a CUDA Graph.

Needs an NVIDIA GPU (compute capability 6.0 or newer, for double atomics) and CuPy, e.g.
`pip install cupy-cuda12x`. No nvcc is needed.
"""

from __future__ import annotations

import numpy as np

from .kernel_engine import KERNELS, KernelEngine

BLOCK = 256


class CUDAEngine(KernelEngine):
    def __init__(self, *args, device: int = 0, **kwargs):
        try:
            import cupy as cp  # noqa: PLC0415
        except ImportError as e:
            raise ImportError("the CUDA engine needs CuPy: pip install cupy-cuda12x") from e
        self.cp = cp
        cp.cuda.Device(device).use()
        real = {"float32": "float", "float64": "double"}[np.dtype(kwargs.get("dtype", np.float32)).name]
        module = cp.RawModule(code=KERNELS.read_text(), options=("--std=c++14", "--fmad=false", f"-DREAL={real}"))
        self._kernels = {k: module.get_function(k) for k in ("update", "deliver", "clear", "advance")}
        self._stream = cp.cuda.Stream()  # blocking stream: ordered with CuPy's default-stream copies
        self._graphs: dict[int, object] = {}
        sms = cp.cuda.Device(device).attributes["MultiProcessorCount"]
        super().__init__(*args, **kwargs)
        self.update_threads = -(-self.n // BLOCK) * BLOCK
        self.deliver_threads = 2 * sms * BLOCK

    def _array(self, x: np.ndarray):
        return self.cp.asarray(x)

    def _numpy(self, x) -> np.ndarray:
        return self.cp.asnumpy(x)

    def _set(self, dst, src: np.ndarray) -> None:
        dst.set(np.ascontiguousarray(src))

    def _zero(self, x) -> None:
        x.fill(0)

    def _launch(self, kernel: str, threads: int, args: tuple) -> None:
        block = min(BLOCK, threads)
        self._kernels[kernel]((-(-threads // block),), (block,), args)

    def _run_chunk(self, steps: int) -> None:
        graph = self._graphs.get(steps)
        if graph is None:
            with self._stream:
                self._stream.begin_capture()
                self._launch_chunk(steps)
                graph = self._stream.end_capture()
            self._graphs[steps] = graph
        graph.launch(self._stream)
        self._stream.synchronize()

    def warmup(self, n_steps: int) -> None:
        """Compile kernels and capture the graphs a run of n_steps will use, then reset."""
        for k in {min(n_steps, self.chunk_steps), n_steps % self.chunk_steps} - {0}:
            self._run_chunk(k)
        self.reset()
