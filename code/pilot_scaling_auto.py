#!/usr/bin/env python3
"""
Run time and ARI of the AUTOMATIC FLOPTICS-SA and label-free baselines as N grows.

Data: the labelled 2D and 3D mixtures of pilot_grid_offset.py (5% uniform noise).
Every run happens in its own process and is stopped after --time-budget seconds; a method
that times out is skipped for larger N.

Methods
  floptics_auto     full automatic method (chooses m itself every time)
  floptics_sa_fixed one run at the m that floptics_auto chose at the smallest N
                    (cost of clustering once m is known)
  hdbscan_1pct, dbscan_knee, gmm_bic, optics_xi   (see baselines_free.py)
  floptics_b20, floptics_b50   original FLOPTICS (Bin_size 20 / 50, MinPts 10)

Usage
  python pilot_scaling_auto.py
  python pilot_scaling_auto.py --sizes 10000,100000 --time-budget 120
"""

import argparse
import json
import multiprocessing as mp
import queue
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pilot_grid_offset import make_scaling_data, parse_list  # noqa: E402
from auto_core import floptics_auto, floptics_sa_sig  # noqa: E402
from baselines_free import dbscan_knee, gmm_bic, optics_xi  # noqa: E402

warnings.filterwarnings("ignore")
METHODS = ["floptics_auto", "floptics_sa_fixed", "hdbscan_1pct", "dbscan_knee", "gmm_bic",
           "optics_xi", "automato", "floptics_b20", "floptics_b50"]
COL = {"floptics_auto": "#1b9e77", "floptics_sa_fixed": "#66c2a5", "hdbscan_1pct": "#e7298a",
       "dbscan_knee": "#7570b3", "gmm_bic": "#d95f02", "optics_xi": "#a6761d",
       "automato": "#e6ab02", "floptics_b20": "#984ea3", "floptics_b50": "#377eb8"}


def run_one(method, X, m_fixed):
    N, d = X.shape
    if method == "floptics_auto":
        return floptics_auto(X, np.zeros(d))[0]
    if method == "floptics_sa_fixed":
        return floptics_sa_sig(X, m_fixed, 4, np.zeros(d), 1.0, 1.0)
    if method == "hdbscan_1pct":
        return HDBSCAN(min_cluster_size=max(5, int(0.01 * N))).fit_predict(X)
    if method == "dbscan_knee":
        return dbscan_knee(X)
    if method == "gmm_bic":
        return gmm_bic(X)
    if method == "optics_xi":
        return optics_xi(X)
    if method in ("floptics_b20", "floptics_b50"):
        from pilot_floptics import floptics
        from pilot_grid_offset import normalise
        return floptics(normalise(X), 20 if method == "floptics_b20" else 50, 10, np.zeros(d))
    if method == "automato":
        from automato import Automato
        return Automato(random_state=42).fit(X).labels_
    raise ValueError(method)


def _child(method, X, m_fixed, q):
    warnings.filterwarnings("ignore")
    if method.startswith("floptics"):  # compile numba code outside the timed part
        floptics_sa_sig(np.random.rand(50, X.shape[1]), 5, 2, np.zeros(X.shape[1]), 1.0, 1.0)
    t = time.perf_counter()
    lab = run_one(method, X, m_fixed)
    q.put((time.perf_counter() - t, lab))


def timed(method, X, m_fixed, budget):
    q = mp.Queue()
    p = mp.Process(target=_child, args=(method, X, m_fixed, q))
    p.start()
    try:
        secs, lab = q.get(timeout=budget)
    except queue.Empty:
        p.terminate()
        p.join()
        return None, None
    p.join()
    return secs, lab


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="10000,30000,100000,300000,1000000")
    ap.add_argument("--time-budget", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="results_scaling_auto")
    ap.add_argument("--methods", default=",".join(METHODS))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sizes = parse_list(args.sizes, int)
    rows = []
    for d in (2, 3):
        X0, _ = make_scaling_data(sizes[0], d, args.seed)
        _, m_fixed = floptics_auto(X0, np.zeros(d))
        print(f"d={d}: floptics_auto chose m={m_fixed} at n={sizes[0]}")
        stopped = set()
        for n in sizes:
            X, y = make_scaling_data(n, d, args.seed + n)
            for method in [m for m in args.methods.split(",") if m]:
                if method in stopped:
                    rows.append(dict(d=d, n=n, method=method, seconds=np.nan, ari=np.nan,
                                     status="skipped"))
                    continue
                secs, lab = timed(method, X, m_fixed, args.time_budget)
                if secs is None:
                    stopped.add(method)
                    rows.append(dict(d=d, n=n, method=method, seconds=np.nan, ari=np.nan,
                                     status=f"timeout>{args.time_budget}s"))
                    print(f"  d={d} n={n:>9,} {method:18s} TIMEOUT")
                    continue
                ari = adjusted_rand_score(y, lab)
                rows.append(dict(d=d, n=n, method=method, seconds=secs, ari=ari, status="ok",
                                 m_fixed=m_fixed))
                print(f"  d={d} n={n:>9,} {method:18s} {secs:9.3f}s  ARI={ari:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "scaling_auto.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, d in zip(axes, (2, 3)):
        for mt in df.method.unique():
            s = df[(df.d == d) & (df.method == mt) & (df.status == "ok")]
            if len(s):
                ax.loglog(s.n, s.seconds, "-o", c=COL[mt], label=mt)
        ax.set_xlabel("number of points N")
        ax.set_ylabel("seconds")
        ax.set_title(f"d={d} (timeout {args.time_budget}s; missing points = timed out)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "scaling_auto.png", dpi=150)
    print(json.dumps({"saved": str(out.resolve())}))


if __name__ == "__main__":
    main()
