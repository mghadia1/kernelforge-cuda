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
SIM_LIB = REPO_ROOT / "build" / "libkfsim.so"

VERSIONS = ("v0_naive", "v1_shared", "v2_warp", "v3_topk", "v4_batch",
            "v5_regblock")
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


# --- CPU simulation of v3's selection ---------------------------------------
#
# selection_sim.cpp runs v3's two-stage top-k serially with no CUDA, so the part
# of v3 whose answer depends on the work split can be tested on a machine with
# no GPU. It builds in under a second with any host C++ compiler, so the tests
# build it on demand rather than requiring a separate make step.

_sim = None
_sim_error: str | None = None


def build_sim(force: bool = False) -> bool:
    """Compile build/libkfsim.so. Returns False (with a reason) if it cannot."""
    global _sim_error
    if SIM_LIB.exists() and not force:
        return True
    import shutil
    import subprocess

    cxx = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
    if cxx is None:
        _sim_error = "no host C++ compiler found (set CXX)"
        return False
    SIM_LIB.parent.mkdir(parents=True, exist_ok=True)
    cmd = [cxx, "-O2", "-std=c++17", "-shared", "-fPIC",
           "-I", str(REPO_ROOT / "src"),
           str(REPO_ROOT / "src" / "selection_sim.cpp"), "-o", str(SIM_LIB)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        _sim_error = f"compiling selection_sim.cpp failed: {proc.stderr.strip()[:400]}"
        return False
    return True


def sim_available() -> bool:
    return _load_sim() is not None


def sim_error() -> str:
    _load_sim()
    return _sim_error or ""


def _load_sim():
    global _sim, _sim_error
    if _sim is not None:
        return _sim
    if _sim_error is not None:
        return None
    if not build_sim():
        return None
    lib = ctypes.CDLL(str(SIM_LIB))
    lib.kf_sim_v3_select.argtypes = [
        np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS"),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"),
    ]
    lib.kf_sim_v3_select.restype = ctypes.c_int
    lib.kf_sim_v4_select.argtypes = lib.kf_sim_v3_select.argtypes
    lib.kf_sim_v4_select.restype = ctypes.c_int
    lib.kf_sim_insert_rule.argtypes = [ctypes.c_float, ctypes.c_int,
                                       ctypes.c_float, ctypes.c_int]
    lib.kf_sim_insert_rule.restype = ctypes.c_int
    _sim = lib
    return _sim


def sim_select(scores: np.ndarray, k: int, version: str = "v3"):
    """Run v3's or v4's decomposition over a (B, N) score matrix.

    Raises RuntimeError if the simulation is unavailable, and ValueError for a k
    the kernel itself would reject, so the test sees the same guard the GPU path
    applies.
    """
    lib = _load_sim()
    if lib is None:
        raise RuntimeError(f"selection simulation unavailable: {sim_error()}")
    scores = np.ascontiguousarray(scores, dtype=np.float32)
    b, n = scores.shape
    vals = np.zeros((b, k), dtype=np.float32)
    idx = np.zeros((b, k), dtype=np.int32)
    fn = lib.kf_sim_v3_select if version == "v3" else lib.kf_sim_v4_select
    if fn(scores, b, n, k, vals, idx) != 0:
        raise ValueError(f"k={k} is outside the range the kernel supports")
    return vals, idx


def sim_outranks(s: float, i: int, v: float, j: int) -> bool:
    """The shared (score, index) comparison, straight from topk_rule.h."""
    lib = _load_sim()
    if lib is None:
        raise RuntimeError(f"selection simulation unavailable: {sim_error()}")
    return bool(lib.kf_sim_insert_rule(s, i, v, j))


# --- persistent corpus -------------------------------------------------------
#
# The kf_v* entry points above pay for uploading the whole corpus on every call,
# which is the honest cost of a cold call and useless as a model of a running
# service. Corpus keeps X on the device so a query pays only for the query
# vectors in, the kernel, and k results out. See src/kf_persistent.h.

PERSISTENT_IMPLS = {"v3_topk": 0, "v4_batch": 1, "v5_regblock": 2, "cublas": 3}


def _bind_persistent(lib):
    if getattr(lib, "_kf_persistent_bound", False):
        return
    lib.kf_corpus_create.argtypes = [
        np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS"),
        ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
    ]
    lib.kf_corpus_create.restype = ctypes.c_void_p
    lib.kf_corpus_query.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS"),
        ctypes.c_int, ctypes.c_int,
        np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"),
        ctypes.POINTER(KfTiming),
    ]
    lib.kf_corpus_query.restype = ctypes.c_int
    lib.kf_corpus_destroy.argtypes = [ctypes.c_void_p]
    lib.kf_corpus_destroy.restype = None
    lib._kf_persistent_bound = True


class Corpus:
    """A corpus resident on the device. Use as a context manager.

        with runner.Corpus(X) as corpus:
            vals, idx, timing = corpus.query(q, k=5, impl="v5_regblock")
    """

    def __init__(self, x: np.ndarray):
        lib = load()
        if lib is None:
            raise RuntimeError(f"kernelforge library unavailable: {load_error()}")
        _bind_persistent(lib)
        self._lib = lib
        x = np.ascontiguousarray(x, dtype=np.float32)
        self.n, self.d = x.shape
        status = ctypes.c_int(0)
        handle = lib.kf_corpus_create(x, self.n, self.d, ctypes.byref(status))
        if not handle:
            raise RuntimeError(f"kf_corpus_create failed with CUDA error {status.value}")
        self._handle = ctypes.c_void_p(handle)

    def query(self, q: np.ndarray, k: int, impl: str = "v5_regblock"):
        if impl not in PERSISTENT_IMPLS:
            raise ValueError(f"unknown impl {impl!r}; expected one of {sorted(PERSISTENT_IMPLS)}")
        if self._handle is None:
            raise RuntimeError("corpus is closed")
        q = np.ascontiguousarray(q, dtype=np.float32)
        b, d = q.shape
        if d != self.d:
            raise ValueError(f"dimension mismatch: corpus d={self.d}, query d={d}")
        vals = np.zeros((b, k), dtype=np.float32)
        idx = np.zeros((b, k), dtype=np.int32)
        timing = KfTiming()
        rc = self._lib.kf_corpus_query(self._handle, PERSISTENT_IMPLS[impl], q, b, k,
                                       vals, idx, ctypes.byref(timing))
        if rc != 0:
            raise RuntimeError(f"kf_corpus_query({impl}) failed with error {rc}")
        return vals, idx, Timing(timing.h2d_ms, timing.kernel_ms, timing.d2h_ms,
                                 timing.host_topk_ms, timing.total_ms)

    def close(self):
        if getattr(self, "_handle", None) is not None:
            self._lib.kf_corpus_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
