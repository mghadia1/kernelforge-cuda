#!/usr/bin/env python3
"""Benchmark sweep: every kernel version against CPU NumPy, PyTorch, and cuBLAS.

Discipline (the same used in the SensorGuard and LinkForge benchmarks):
  * warm up before timing, and discard the first timed run outright - it pays
    for allocation, JIT, and cuBLAS handle setup, and reporting it would flatter
    whichever implementation happened to run second;
  * >= 15 repeats, report median and p95, never the minimum;
  * every implementation gets identical inputs and identical timing points
    (end-to-end wall time for the whole call, transfers included), because a
    kernel-only number would hide the PCIe copy that v3 exists to remove;
  * every row records the device, so results from two machines never get mixed.

Writes a tidy CSV; bench/RESULTS.md interprets it.

Usage:
    python bench/run.py --out bench/results.csv
    python bench/run.py --n 2039,100000 --b 1,32 --repeats 25
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import platform
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import reference  # noqa: E402
import runner  # noqa: E402

DEFAULT_N = [2039, 10000, 100000, 1000000]
DEFAULT_B = [1, 8, 32]
K = 5


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None


def cpu_numpy(q, x, k):
    """CPU baseline: the same thing PaperTrail does today, in NumPy."""
    return reference.reference_topk(q, x, k)


def make_torch_fn(torch):
    def torch_gpu(q, x, k):
        dq = torch.from_numpy(q).cuda()
        dx = torch.from_numpy(x).cuda()
        scores = dq @ dx.T
        vals, idx = torch.topk(scores, k, dim=1)
        out = vals.cpu().numpy(), idx.cpu().numpy().astype(np.int32)
        torch.cuda.synchronize()
        return out
    return torch_gpu


def time_call(fn, repeats, warmup=3):
    """Return (median_ms, p95_ms, n_timed). The first timed run is discarded."""
    for _ in range(warmup):
        fn()
    samples = []
    for i in range(repeats + 1):
        t0 = time.perf_counter()
        fn()
        dt = (time.perf_counter() - t0) * 1000.0
        if i > 0:  # drop the first timed run
            samples.append(dt)
    samples.sort()
    p95 = samples[min(len(samples) - 1, int(round(0.95 * (len(samples) - 1))))]
    return statistics.median(samples), p95, len(samples)


def check(name, got, expected, n):
    """Every timed implementation is verified once at each size, so a fast wrong
    answer can never be reported as a speedup."""
    gv, gi = got
    ev, ei = expected
    max_err = float(np.max(np.abs(np.asarray(gv, dtype=np.float64) - ev)))
    idx_match = bool(np.array_equal(np.asarray(gi, dtype=np.int32), ei))
    if max_err >= 1e-4 or not idx_match:
        print(f"  !! {name} disagrees with reference at N={n} "
              f"(max_err={max_err:.2e}, indices_match={idx_match})")
    return max_err, idx_match


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default=",".join(str(v) for v in DEFAULT_N))
    ap.add_argument("--b", default=",".join(str(v) for v in DEFAULT_B))
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="bench/results.csv")
    ap.add_argument("--impls", default="",
                    help="comma-separated subset of implementations to time")
    ap.add_argument("--skip-cpu-above", type=int, default=200000,
                    help="skip the NumPy baseline above this N (it gets slow)")
    ap.add_argument("--profile-once", action="store_true",
                    help="run one implementation once at one size and exit; for `ncu`")
    ap.add_argument("--profile-impl", default="v3_topk",
                    help="which implementation --profile-once runs")
    args = ap.parse_args()

    ns = [int(v) for v in args.n.split(",")]
    bs = [int(v) for v in args.b.split(",")]

    gpu = runner.available()
    device = runner.device_name() if gpu else f"CPU only ({platform.processor() or platform.machine()})"
    torch = _torch()

    if args.profile_once:
        q, x = reference.make_data(ns[0], bs[0], seed=args.seed)
        runner.run(args.profile_impl, q, x, args.k)
        return 0

    print(f"device: {device}")
    if not gpu:
        print(f"note: {runner.load_error() or 'no CUDA device'}; "
              "GPU rows will be skipped and only the CPU baseline is measured.")
    if torch is None:
        print("note: torch with CUDA not importable; the torch row is skipped.")

    rows = []
    for n in ns:
        for b in bs:
            q, x = reference.make_data(n, b, seed=args.seed)
            expected = reference.reference_topk(q, x, args.k)
            print(f"\nN={n} B={b} d={reference.DIM} k={args.k}")

            impls = []
            if n <= args.skip_cpu_above:
                impls.append(("cpu_numpy", lambda: cpu_numpy(q, x, args.k)))
            if gpu:
                wanted = args.impls.split(",") if args.impls else runner.ALL_IMPLS
                impls += [(v, (lambda v=v: runner.run(v, q, x, args.k)[:2]))
                          for v in runner.ALL_IMPLS if v in wanted]
            if torch is not None:
                tfn = make_torch_fn(torch)
                impls.append(("torch_gpu", lambda tfn=tfn: tfn(q, x, args.k)))

            for name, fn in impls:
                max_err, idx_match = check(name, fn(), expected, n)
                median, p95, count = time_call(fn, args.repeats)
                print(f"  {name:<10} median {median:9.3f} ms   p95 {p95:9.3f} ms")
                rows.append({
                    "impl": name, "N": n, "B": b, "d": reference.DIM, "k": args.k,
                    "median_ms": round(median, 4), "p95_ms": round(p95, 4),
                    "runs": count, "max_abs_err": f"{max_err:.3e}",
                    "indices_match": idx_match, "device": device, "seed": args.seed,
                })

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
