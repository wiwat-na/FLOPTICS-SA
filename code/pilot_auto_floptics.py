#!/usr/bin/env python3
"""
Fully automatic FLOPTICS-SA vs other methods with NO per-dataset tuning.

Every method runs with ONE fixed setting for all datasets:
  floptics_auto   FLOPTICS-SA, k=4, z=1, floor=1 point per coarse bin, m chosen by
                  stability across resolutions at fixed internal offsets, so m depends on
                  the data only (see auto_core.py). No labels used.
  floptics_auto_k1  the same automatic procedure on ONE grid (ablation: no averaging)
  floptics        original FLOPTICS (re-implementation), Bin_size=20, MinPts=10
  grid_averaged   averaged grid + fixed threshold, m=15, alpha=1
  floptics_auto_v1  previous version: m re-chosen at every grid offset
  hdbscan_default HDBSCAN, scikit-learn defaults (min_cluster_size=5)
  hdbscan_1pct    HDBSCAN, min_cluster_size = 1% of N
  optics_xi, gmm_bic, dbscan_knee, dpc: label-free baselines, see baselines_free.py
  kde_auto, walther_kde: FLOPTICS-SA with a binned Gaussian KDE instead of ASH (kde_variants.py)
  automato        AuToMATo (Huber, Kalisnik & Schnider 2024), package defaults, random_state=42
The fixed settings of floptics, grid_averaged and the z / m-rule of floptics_auto were
chosen on the DEVELOPMENT datasets. The TEST datasets were not looked at before this run.

Grid methods run on several random grid offsets; HDBSCAN once (it uses no grid).

Usage
  python pilot_auto_floptics.py --data-dir clustering-data-v1
  python pilot_auto_floptics.py --data-dir clustering-data-v1 --quick
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pilot_grid_offset import load_benchmark, normalise  # noqa: E402
from pilot_floptics import floptics, grid_averaged, paper_dataset, scores  # noqa: E402
from auto_core import floptics_auto  # noqa: E402
from baselines_free import dbscan_knee, dpc, gmm_bic, optics_xi  # noqa: E402
from kde_variants import kde_auto, walther_kde  # noqa: E402

warnings.filterwarnings("ignore")

DEV = ["paper/A", "paper/B", "paper/C", "paper/D",
       "fcps/atom", "fcps/chainlink", "fcps/hepta", "fcps/lsun", "fcps/target", "fcps/tetra",
       "other/hdbscan", "sipu/aggregation", "sipu/compound", "sipu/d31", "sipu/flame",
       "sipu/jain", "sipu/pathbased", "sipu/r15", "sipu/s1", "sipu/spiral", "wut/smile"]
TEST = ["wut/circles", "wut/cross", "wut/graph", "wut/labirynth", "wut/mk1", "wut/mk2",
        "wut/mk3", "wut/mk4", "wut/olympic", "wut/stripes", "wut/trapped_lovers",
        "wut/twosplashes", "wut/windows", "wut/x1", "wut/x2", "wut/x3", "wut/z1", "wut/z2",
        "wut/z3", "graves/dense", "graves/fuzzyx", "graves/line", "graves/parabolic",
        "graves/ring", "graves/ring_noisy", "graves/ring_outliers", "graves/zigzag",
        "graves/zigzag_noisy", "graves/zigzag_outliers", "sipu/a1", "sipu/a2", "sipu/s2",
        "sipu/s3", "sipu/s4", "fcps/engytime", "fcps/twodiamonds", "fcps/wingnut",
        "other/square"]
METHODS = ["floptics_auto", "floptics_auto_v1", "floptics_auto_k1", "floptics", "grid_averaged",
           "hdbscan_default", "hdbscan_1pct", "optics_xi", "gmm_bic", "dbscan_knee", "dpc",
           "kde_auto", "walther_kde", "automato"]
DETERMINISTIC = {"hdbscan_default", "hdbscan_1pct", "optics_xi", "gmm_bic", "dbscan_knee", "dpc",
                 "automato"}


def load(name, data_dir, seed):
    if name.startswith("paper/"):
        return paper_dataset(name[-1], seed)
    return load_benchmark(data_dir, name)


def run_method(method, Xn, s):
    N, d = Xn.shape
    if method == "floptics_auto":
        return floptics_auto(Xn, s, k=4)            # m chosen from the data only
    if method == "floptics_auto_v1":
        return floptics_auto(Xn, s, k=4, m_rule="stable")   # previous version
    if method == "floptics_auto_k1":
        return floptics_auto(Xn, s, k=1)            # ablation: no averaging
    if method == "floptics":
        return floptics(Xn, 20, 10, s), 20
    if method == "grid_averaged":
        return grid_averaged(Xn, 15, 1.0, 4, s), 15
    if method == "hdbscan_default":
        return HDBSCAN().fit_predict(Xn), np.nan
    if method == "hdbscan_1pct":
        return HDBSCAN(min_cluster_size=max(5, int(0.01 * N))).fit_predict(Xn), np.nan
    if method == "kde_auto":
        return kde_auto(Xn, s)
    if method == "walther_kde":
        return walther_kde(Xn, s), np.nan
    if method == "automato":
        from automato import Automato
        return Automato(random_state=42).fit(Xn).labels_, np.nan
    if method == "optics_xi":
        return optics_xi(Xn), np.nan
    if method == "gmm_bic":
        return gmm_bic(Xn), np.nan
    if method == "dbscan_knee":
        return dbscan_knee(Xn), np.nan
    if method == "dpc":
        lab = dpc(Xn)
        return (lab if lab is not None else np.full(N, -1)), np.nan
    raise ValueError(method)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--runs", type=int, default=10, help="grid offsets per dataset")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default="results_auto")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--methods", default=",".join(METHODS))
    args = p.parse_args()
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    sets = {"dev": DEV[:4] + DEV[10:12], "test": TEST[:6]} if args.quick else {"dev": DEV, "test": TEST}
    runs = 3 if args.quick else args.runs
    floptics_auto(np.random.rand(60, 2), np.zeros(2))  # compile once
    rows = []
    t0 = time.time()
    for split, names in sets.items():
        print(f"== {split} ==")
        for nm in names:
            res = load(nm, args.data_dir, args.seed)
            if res is None:
                print(f"  [missing] {nm}")
                continue
            X, y = res
            Xn = normalise(X)
            offs = rng.rand(runs, Xn.shape[1])
            t1 = time.time()
            for method in [mm for mm in args.methods.split(",") if mm]:
                n_runs = 1 if method in DETERMINISTIC else runs
                for r in range(n_runs):
                    t = time.perf_counter()
                    lab, m = run_method(method, Xn, offs[r])
                    rows.append(dict(split=split, dataset=nm, n=len(y), d=Xn.shape[1],
                                     method=method, run=r, m=m,
                                     seconds=time.perf_counter() - t, **scores(y, lab)))
            print(f"  {nm:24s} n={len(y):5d} {time.time() - t1:6.1f}s")
    df = pd.DataFrame(rows)
    df.to_csv(out / "auto_runs.csv", index=False)

    agg = (df.groupby(["split", "dataset", "method"])
             .agg(ari_mean=("ari", "mean"), ari_std=("ari", "std"), sec=("seconds", "mean"),
                  m_min=("m", "min"), m_max=("m", "max"))
             .reset_index())
    agg.to_csv(out / "auto_summary.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    for split in sets:
        tab = agg[agg.split == split].pivot(index="dataset", columns="method", values="ari_mean")[METHODS]
        std = agg[agg.split == split].pivot(index="dataset", columns="method", values="ari_std")[METHODS]
        sec = agg[agg.split == split].pivot(index="dataset", columns="method", values="sec")[METHODS]
        print(f"\n=== {split.upper()} set: mean ARI (one fixed setting per method) ===")
        print(tab.round(3).to_string())
        print("median ARI : " + "  ".join(f"{m}={tab[m].median():.3f}" for m in METHODS))
        print("mean ARI   : " + "  ".join(f"{m}={tab[m].mean():.3f}" for m in METHODS))
        print("median std : " + "  ".join(f"{m}={std[m].median():.3f}" for m in METHODS
                                         if m not in DETERMINISTIC))
        print("median sec : " + "  ".join(f"{m}={sec[m].median():.4f}" for m in METHODS))
        for other in METHODS[1:]:
            dlt = tab["floptics_auto"] - tab[other]
            print(f"floptics_auto vs {other:16s}: higher {int((dlt > 0.01).sum()):2d}  "
                  f"lower {int((dlt < -0.01).sum()):2d}  tie {int((dlt.abs() <= 0.01).sum()):2d}")
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.boxplot([tab[m].dropna().values for m in METHODS], patch_artist=True)
        ax.set_xticks(range(1, len(METHODS) + 1))
        ax.set_xticklabels(METHODS, rotation=30, ha="right")
        ax.set_ylabel("mean ARI per dataset")
        ax.set_title(f"{split} set ({len(tab)} datasets), one fixed setting per method")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "figures" / f"auto_{split}_boxplot.png", dpi=150)
        plt.close(fig)
    print(f"\nTotal time {time.time() - t0:.1f}s. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
