#!/usr/bin/env python3
"""Warm-service benchmark: the corpus is uploaded once, then queried.

bench/run.py measures a cold call — allocate, upload the whole corpus, score,
select, free. That is an honest number and a useless model of a retrieval
service, because at N = 1M the upload is ~350 ms and the scoring kernel is under
4 ms. Every kernel improvement disappears into the transfer.

This script asks the other question: once PaperTrail has its embeddings on the
GPU, what does a query cost? Same discipline as run.py — warm up, discard the
first timed run, >= 15 repeats, median and p95, verify against the NumPy
reference before timing, record the device.

The CPU baseline needs no special treatment: NumPy's corpus is already resident
in RAM, so `cpu_numpy` was always measuring the warm case. torch keeps X on the
device across queries here, matching what it would do in production.

Usage:
    python bench/persistent.py --out bench/results_persistent.csv
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


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None


def time_call(fn, repeats, warmup=3):
    """(median_ms, p95_ms, n). The first timed run is discarded."""
    for _ in range(warmup):
        fn()
    samples = []
    for i in range(repeats + 1):
        t0 = time.perf_counter()
        fn()
        dt = (time.perf_counter() - t0) * 1000.0
        if i > 0:
            samples.append(dt)
    samples.sort()
    p95 = samples[min(len(samples) - 1, int(round(0.95 * (len(samples) - 1))))]
    return statistics.median(samples), p95, len(samples)


def check(name, got, expected, n):
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
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="bench/results_persistent.csv")
    ap.add_argument("--skip-cpu-above", type=int, default=200000)
    args = ap.parse_args()

    ns = [int(v) for v in args.n.split(",")]
    bs = [int(v) for v in args.b.split(",")]

    gpu = runner.available()
    device = runner.device_name() if gpu else f"CPU only ({platform.processor() or platform.machine()})"
    torch = _torch()
    print(f"device: {device}")
    if not gpu:
        print(f"note: {runner.load_error() or 'no CUDA device'}; GPU rows skipped.")

    rows = []
    for n in ns:
        q_all, x = reference.make_data(n, max(bs), seed=args.seed)
        corpus = runner.Corpus(x) if gpu else None
        tx = None
        if torch is not None:
            tx = torch.from_numpy(x).cuda()      # resident, as a service would
            torch.cuda.synchronize()

        try:
            for b in bs:
                q = np.ascontiguousarray(q_all[:b])
                expected = reference.reference_topk(q, x, args.k)
                print(f"\nN={n} B={b} d={reference.DIM} k={args.k}  (corpus resident)")

                impls = []
                if n <= args.skip_cpu_above:
                    impls.append(("cpu_numpy", lambda: reference.reference_topk(q, x, args.k)))
                if corpus is not None:
                    for name in ("v3_topk", "v4_batch", "v5_regblock", "cublas"):
                        impls.append((name, (lambda name=name: corpus.query(q, args.k, name)[:2])))
                if torch is not None:
                    def torch_warm(q=q):
                        dq = torch.from_numpy(q).cuda()
                        vals, idx = torch.topk(dq @ tx.T, args.k, dim=1)
                        out = vals.cpu().numpy(), idx.cpu().numpy().astype(np.int32)
                        torch.cuda.synchronize()
                        return out
                    impls.append(("torch_gpu", torch_warm))

                for name, fn in impls:
                    max_err, idx_match = check(name, fn(), expected, n)
                    median, p95, count = time_call(fn, args.repeats)
                    print(f"  {name:<12} median {median:9.3f} ms   p95 {p95:9.3f} ms")
                    rows.append({
                        "mode": "persistent", "impl": name, "N": n, "B": b,
                        "d": reference.DIM, "k": args.k,
                        "median_ms": round(median, 4), "p95_ms": round(p95, 4),
                        "runs": count, "max_abs_err": f"{max_err:.3e}",
                        "indices_match": idx_match, "device": device, "seed": args.seed,
                    })
        finally:
            if corpus is not None:
                corpus.close()
            if tx is not None:
                del tx
                torch.cuda.empty_cache()

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
