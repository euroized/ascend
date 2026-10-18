"""Record the hardware and software a measurement ran on. Every published number carries this."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata

from . import data

PACKAGES = ["numpy", "pandas", "pyarrow", "brian2", "cython", "torch", "cupy", "triton"]


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def cpu_model() -> str:
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if sys.platform == "darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    return platform.processor() or platform.machine()


def ram_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    if sys.platform == "darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        return int(out) if out else None
    return None


def gpus() -> list[dict]:
    """NVIDIA GPUs via nvidia-smi; empty list when none are visible."""
    if not shutil.which("nvidia-smi"):
        return []
    q = "name,driver_version,memory.total,compute_cap,pcie.link.gen.max,pcie.link.width.max"
    out = _run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"])
    if not out:
        return []
    keys = ["name", "driver", "memory_mib", "compute_capability", "pcie_gen", "pcie_width"]
    return [dict(zip(keys, (v.strip() for v in line.split(",")))) for line in out.splitlines()]


def cuda_runtime() -> str | None:
    try:
        import torch  # noqa: PLC0415

        return torch.version.cuda if torch.cuda.is_available() else None
    except ImportError:
        return None


def packages() -> dict[str, str]:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


def git_state() -> dict:
    root = str(data.REPO_ROOT)
    commit = _run(["git", "-C", root, "rev-parse", "HEAD"])
    dirty = _run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"])
    return {"commit": commit, "dirty": bool(dirty) if dirty is not None else None}


def info() -> dict:
    ram = ram_bytes()
    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu_model(),
        "cpu_logical_cores": os.cpu_count(),
        "ram_gb": round(ram / 2**30, 1) if ram else None,
        "gpus": gpus(),
        "cuda_runtime": cuda_runtime(),
        "python": platform.python_version(),
        "packages": packages(),
        "repo": git_state(),
    }
