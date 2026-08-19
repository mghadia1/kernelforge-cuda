"""ctypes bridge to the compiled CUDA kernels.

The .cu files are built into a single shared library by the Makefile
(``make``), which this module loads lazily. Everything here degrades quietly
when there is no GPU: ``available()`` returns False, and the tests and
benchmark skip the GPU paths instead of failing. That is what lets CI run the
NumPy-reference checks on a CPU-only runner while the real numbers come from a
Colab/Kaggle T4.
"""

from __future__ import annotations

import ctypes
import os
import pathlib
from dataclasses import dataclass

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_LIB = REPO_ROOT / "build" / "libkernelforge.so"

VERSIONS = ("v0_naive", "v1_shared", "v2_warp", "v3_topk")
BASELINES = ("cublas",)  # library reference, not a hand-written kernel
ALL_IMPLS = VERSIONS + BASELINES


class KfTiming(ctypes.Structure):
    _fields_ = [
        ("h2d_ms", ctypes.c_float),
        ("kernel_ms", ctypes.c_float),
        ("d2h_ms", ctypes.c_float),
        ("host_topk_ms", ctypes.c_float),
        ("total_ms", ctypes.c_float),
    ]


@dataclass(frozen=True)
class Timing:
    h2d_ms: float
    kernel_ms: float
    d2h_ms: float
    host_topk_ms: float
    total_ms: float


_lib = None
_load_error: str | None = None


def library_path() -> pathlib.Path:
    return pathlib.Path(os.environ.get("KERNELFORGE_LIB", str(DEFAULT_LIB)))


def load():
    """Return the loaded library, or None with the reason recorded."""
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib

    path = library_path()
    if not path.exists():
        _load_error = f"{path} not built (run `make`)"
        return None
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as exc:  # no CUDA runtime on this box, wrong arch, ...
        _load_error = f"cannot load {path}: {exc}"
        return None

    sig = [
        np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS"),  # q
        np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS"),  # X
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,    # B, N, d, k
        np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS"),  # out_vals
        np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"),    # out_idx
        ctypes.POINTER(KfTiming),
    ]
    for name in ALL_IMPLS:
        fn = getattr(lib, f"kf_{name}")
        fn.argtypes = sig
        fn.restype = ctypes.c_int

    lib.kf_device_count.argtypes = []
    lib.kf_device_count.restype = ctypes.c_int
    lib.kf_device_name.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.kf_device_name.restype = ctypes.c_int

    _lib = lib
    return _lib


def load_error() -> str:
    load()
    return _load_error or ""


def available() -> bool:
    """True only when the library loads *and* a CUDA device is present."""
    lib = load()
    if lib is None:
        return False
    try:
        return lib.kf_device_count() > 0
    except OSError:
        return False


def device_name() -> str:
    lib = load()
    if lib is None:
        return "no CUDA library"
    buf = ctypes.create_string_buffer(256)
    if lib.kf_device_name(buf, len(buf)) != 0:
        return "unknown CUDA device"
    return buf.value.decode()


def run(version: str, q: np.ndarray, x: np.ndarray, k: int):
    """Run one kernel version. Returns (values, indices, Timing).

    Raises RuntimeError if the library is missing or the kernel reports a CUDA
    error, so a silent wrong answer is never mistaken for a result.
    """
    if version not in ALL_IMPLS:
        raise ValueError(f"unknown version {version!r}; expected one of {ALL_IMPLS}")
    lib = load()
    if lib is None:
        raise RuntimeError(f"kernelforge library unavailable: {load_error()}")

    q = np.ascontiguousarray(q, dtype=np.float32)
    x = np.ascontiguousarray(x, dtype=np.float32)
    b, d = q.shape
    n, dx = x.shape
    if d != dx:
        raise ValueError(f"dimension mismatch: q has d={d}, X has d={dx}")

    vals = np.zeros((b, k), dtype=np.float32)
    idx = np.zeros((b, k), dtype=np.int32)
    timing = KfTiming()

    rc = getattr(lib, f"kf_{version}")(q, x, b, n, d, k, vals, idx,
                                       ctypes.byref(timing))
    if rc != 0:
        raise RuntimeError(f"kf_{version} failed with CUDA error {rc}")

    return vals, idx, Timing(timing.h2d_ms, timing.kernel_ms, timing.d2h_ms,
                             timing.host_topk_ms, timing.total_ms)
