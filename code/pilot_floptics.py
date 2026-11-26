#!/usr/bin/env python3
"""
Does FLOPTICS (Sriphum, Wills & Green) depend on where the grid is placed?

FLOPTICS, as described in the paper:
  1. normalise the data and cut every axis into Bin_size equal bins
  2. keep only the non-empty bins; each bin is represented by its integer coordinates
  3. run OPTICS on the non-empty bins (distance = Euclidean distance between bin
     coordinates, a bin is a core-bin if it has >= MinPts neighbouring bins)
  4. every data point takes the label of its bin
Paper settings: Bin_size = 100, MinPts = 10, epsilon chosen automatically by OPTICS.

RE-IMPLEMENTATION NOTES (choices the paper does not fix; please check against the
original R code):
  * OPTICS from scikit-learn, max_eps = infinity, clusters extracted with the
    automatic "xi" method (xi = 0.05, scikit-learn default).
  * Bins are not weighted by how many points they hold (the paper counts neighbouring
    bins, not points).

What this script measures: the grid is shifted by a random fraction of one bin
(different on each axis) and FLOPTICS is run again. If the result changes, FLOPTICS
is sensitive to grid placement. The grid-averaged method from pilot_grid_offset.py
is run on the same offsets for comparison.

Experiments
  P: the four datasets A-D of the FLOPTICS paper (Table 5), rebuilt from the
     published means and covariances (10,000 points, 3 Gaussian clusters).
  Q: the same 2D/3D benchmark datasets as pilot_grid_offset.py (optional).

Usage
  python pilot_floptics.py --quick
  python pilot_floptics.py                                   # experiment P only
  python pilot_floptics.py --data-dir clustering-data-v1     # P and Q
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS
from sklearn.metrics import adjusted_rand_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pilot_grid_offset import (DEFAULT_BENCHMARKS, grid_counts, label_points,  # noqa: E402
                               load_benchmark, matched_correct, normalise,
                               parse_list, structure, triangle_smooth)

# ----------------------------------------------------------------------------- data
PAPER_SIGMA = {1: [[6, 15], [15, 120]], 2: [[2, 0.3], [0.3, 5]], 3: [[3, 2], [2, 10]]}
PAPER_N = {1: 5000, 2: 2500, 3: 2500}
PAPER_MEANS = {  # Table 5: cluster -> (x, y) for datasets A, B, C, D
    "A": {1: (5, 35), 2: (-5, -10), 3: (16, 1)},
    "B": {1: (3, 30), 2: (-5, -10), 3: (12, 1)},
    "C": {1: (1, 25), 2: (-5, -10), 3: (8, 1)},
    "D": {1: (-1, 20), 2: (-5, -10), 3: (4, 1)},
}


def paper_dataset(name, seed):
    rng = np.random.RandomState(seed)
    X, y = [], []
    for c in (1, 2, 3):
        X.append(rng.multivariate_normal(PAPER_MEANS[name][c], PAPER_SIGMA[c], PAPER_N[c]))
        y.append(np.full(PAPER_N[c], c))
    return np.vstack(X), np.concatenate(y)


# ----------------------------------------------------------------------------- methods
def floptics(Xn, bins, min_pts, offset, xi=0.05):
    """FLOPTICS on a grid shifted by `offset` (fraction of one bin per axis)."""
    idx = np.floor(Xn * bins + offset).astype(np.int64)
    occupied, inverse = np.unique(idx, axis=0, return_inverse=True)
    if len(occupied) <= min_pts:
        return np.full(len(Xn), -1)
    bin_lab = OPTICS(min_samples=min_pts, xi=xi).fit_predict(occupied.astype(float))
    return bin_lab[inverse.ravel()]


def grid_averaged(Xn, m, alpha, k, offset, adjacency="chebyshev"):
    N, d = Xn.shape
    c, f = grid_counts(Xn, m * k, offset * k, pad=k)
    return label_points(triangle_smooth(c, k), f, alpha * N / m ** d, structure(d, adjacency))[0]


def scores(y, lab):
    correct = matched_correct(y, lab)
    # paper metric: accuracy of each reference cluster, then the mean over clusters
    per_cluster = [correct[y == c].mean() for c in np.unique(y)]
    return dict(ari=adjusted_rand_score(y, lab),
                paper_accuracy=100 * float(np.mean(per_cluster)),
                noise_frac=float((lab == -1).mean()),
                n_clusters=len(set(lab.tolist()) - {-1}))


# ----------------------------------------------------------------------------- experiment
def run_offsets(name, X, y, settings_fl, settings_avg, runs, k, rng, include_zero=True):
    Xn = normalise(X)
    d = Xn.shape[1]
    offsets = rng.rand(runs, d)
    if include_zero:
        offsets[0] = 0.0  # run 0 = the unshifted grid, as in the original paper
    rows, labels = [], {}
    for r, s in enumerate(offsets):
        for (bins, mp_) in settings_fl:
            t = time.perf_counter()
            lab = floptics(Xn, bins, mp_, s)
            sec = time.perf_counter() - t
            rows.append(dict(dataset=name, method="floptics", setting=f"bins={bins},MinPts={mp_}",
                             run=r, offset=json.dumps(np.round(s, 4).tolist()), seconds=sec,
                             **scores(y, lab)))
            labels[("floptics", bins, mp_, r)] = lab
        for (m, alpha) in settings_avg:
            t = time.perf_counter()
            lab = grid_averaged(Xn, m, alpha, k, s)
            sec = time.perf_counter() - t
            rows.append(dict(dataset=name, method="grid_averaged",
                             setting=f"m={m},alpha={alpha},k={k}", run=r,
                             offset=json.dumps(np.round(s, 4).tolist()), seconds=sec,
                             **scores(y, lab)))
            labels[("grid_averaged", m, alpha, r)] = lab
    return pd.DataFrame(rows), Xn, labels


def summarise(df):
    g = df.groupby(["dataset", "method", "setting"])
    s = g.agg(ari_mean=("ari", "mean"), ari_std=("ari", "std"), ari_min=("ari", "min"),
              ari_max=("ari", "max"), acc_mean=("paper_accuracy", "mean"),
              acc_std=("paper_accuracy", "std"), acc_min=("paper_accuracy", "min"),
              acc_max=("paper_accuracy", "max"), sec_mean=("seconds", "mean"),
              k_min=("n_clusters", "min"), k_max=("n_clusters", "max")).reset_index()
    s["ari_range"] = s.ari_max - s.ari_min
    s["acc_range"] = s.acc_max - s.acc_min
    zero = (df[df.run == 0].set_index(["dataset", "method", "setting"])
            [["ari", "paper_accuracy"]].rename(columns={"ari": "ari_offset0",
                                                        "paper_accuracy": "acc_offset0"}))
    return s.merge(zero.reset_index(), on=["dataset", "method", "setting"])


def best_per_method(summary):
    return summary.loc[summary.groupby(["dataset", "method"])["ari_mean"].idxmax()]


# ----------------------------------------------------------------------------- figures
def fig_worst_best(name, Xn, y, df, labels, setting, path):
    sub = df[(df.dataset == name) & (df.method == "floptics") & (df.setting == setting)]
    worst, best = sub.loc[sub.ari.idxmin()], sub.loc[sub.ari.idxmax()]
    bins = int(setting.split(",")[0].split("=")[1])
    mp_ = int(setting.split(",")[1].split("=")[1])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, row, title in ((axes[0], best, "best offset"), (axes[1], worst, "worst offset")):
        lab = labels[("floptics", bins, mp_, int(row.run))]
        noise = lab == -1
        ax.scatter(Xn[noise, 0], Xn[noise, 1], s=2, c="lightgrey")
        ax.scatter(Xn[~noise, 0], Xn[~noise, 1], s=2, c=lab[~noise] % 10, cmap="tab10")
        ax.set_title(f"FLOPTICS, {title} {json.loads(row.offset)}\n"
                     f"ARI = {row.ari:.3f}, accuracy = {row.paper_accuracy:.1f}%, "
                     f"{int(row.n_clusters)} clusters", fontsize=10)
        ax.set_aspect("equal")
    fig.suptitle(f"{name}  ({setting})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_box(df, best, path, title):
    names = list(dict.fromkeys(best.dataset))
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(names)), 4.5))
    data, pos, cols = [], [], []
    for i, nm in enumerate(names):
        for j, (method, c) in enumerate((("floptics", "#d95f02"), ("grid_averaged", "#1b9e77"))):
            b = best[(best.dataset == nm) & (best.method == method)]
            if b.empty:
                continue
            v = df[(df.dataset == nm) & (df.method == method) & (df.setting == b.iloc[0].setting)].ari
            data.append(v.values)
            pos.append(i * 3 + j)
            cols.append(c)
    bp = ax.boxplot(data, positions=pos, widths=0.8, patch_artist=True)
    for p_, c in zip(bp["boxes"], cols):
        p_.set_facecolor(c)
        p_.set_alpha(0.6)
    ax.set_xticks([i * 3 + 0.5 for i in range(len(names))])
    ax.set_xticklabels([n.split("/")[-1] for n in names], rotation=45, ha="right")
    ax.set_ylabel("ARI across random grid offsets")
    ax.set_title(title)
    ax.legend([bp["boxes"][0], bp["boxes"][1]], ["FLOPTICS (single grid)", "grid averaged (k shifts)"],
              loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None, help="clustering-data-v1 folder (experiment Q)")
    p.add_argument("--datasets", default=",".join(DEFAULT_BENCHMARKS))
    p.add_argument("--runs", type=int, default=50, help="random offsets for experiment P")
    p.add_argument("--runs-q", type=int, default=20, help="random offsets for experiment Q")
    p.add_argument("--paper-bins", type=int, default=100)
    p.add_argument("--paper-minpts", type=int, default=10)
    p.add_argument("--q-bins", default="20,30,50", help="Bin_size values tried in experiment Q")
    p.add_argument("--q-minpts", default="5,10")
    p.add_argument("--m", default="10,15,20,30", help="grid-averaged: bins per axis")
    p.add_argument("--alpha", default="1,2,4", help="grid-averaged: threshold multiples")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--max-n", type=int, default=6000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default="results_floptics")
    p.add_argument("--quick", action="store_true", help="datasets A and D only, 10 offsets")
    args = p.parse_args()

    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    avg_settings = [(m, a) for m in parse_list(args.m, int) for a in parse_list(args.alpha, float)]
    names = ["A", "D"] if args.quick else ["A", "B", "C", "D"]
    runs = 10 if args.quick else args.runs
    t0 = time.time()

    # ---------------- experiment P
    print(f"Experiment P: FLOPTICS paper datasets, Bin_size={args.paper_bins}, "
          f"MinPts={args.paper_minpts}, {runs} offsets (run 0 = unshifted grid)")
    allP = []
    for nm in names:
        X, y = paper_dataset(nm, args.seed)
        t1 = time.time()
        df, Xn, labels = run_offsets(f"paper/{nm}", X, y, [(args.paper_bins, args.paper_minpts)],
                                     avg_settings, runs, args.k, rng)
        allP.append(df)
        fig_worst_best(f"paper/{nm}", Xn, y, df, labels,
                       f"bins={args.paper_bins},MinPts={args.paper_minpts}",
                       out / "figures" / f"floptics_paper_{nm}_worst_best.png")
        print(f"  dataset {nm}: {time.time() - t1:.1f}s")
    dfP = pd.concat(allP, ignore_index=True)
    dfP.to_csv(out / "P_runs.csv", index=False)
    sP = summarise(dfP)
    sP.to_csv(out / "P_summary_all.csv", index=False)
    bP = best_per_method(sP)
    bP.to_csv(out / "P_summary_best.csv", index=False)
    fig_box(dfP, bP, out / "figures" / "P_boxplot.png",
            "FLOPTICS paper datasets: sensitivity to grid placement")
    cols = ["dataset", "method", "setting", "acc_offset0", "acc_mean", "acc_std", "acc_min",
            "acc_max", "ari_offset0", "ari_mean", "ari_std", "ari_min", "ari_max", "k_min",
            "k_max", "sec_mean"]
    pd.set_option("display.width", 250)
    print("\n=== Experiment P (accuracy in % as defined in the FLOPTICS paper) ===")
    print(bP[cols].round(3).to_string(index=False))

    # ---------------- experiment Q
    if args.data_dir and not args.quick:
        fl_settings = [(b, mp_) for b in parse_list(args.q_bins, int)
                       for mp_ in parse_list(args.q_minpts, int)]
        print(f"\nExperiment Q: benchmarks, FLOPTICS settings {fl_settings}, {args.runs_q} offsets")
        allQ = []
        for nm in parse_list(args.datasets, str):
            res = load_benchmark(args.data_dir, nm)
            if res is None or res[0].shape[1] not in (2, 3) or len(res[1]) > args.max_n:
                print(f"  [skip] {nm}")
                continue
            t1 = time.time()
            df, Xn, labels = run_offsets(nm, res[0], res[1], fl_settings, avg_settings,
                                         args.runs_q, args.k, rng)
            allQ.append(df)
            print(f"  {nm:22s} {time.time() - t1:6.1f}s")
        dfQ = pd.concat(allQ, ignore_index=True)
        dfQ.to_csv(out / "Q_runs.csv", index=False)
        sQ = summarise(dfQ)
        sQ.to_csv(out / "Q_summary_all.csv", index=False)
        bQ = best_per_method(sQ)
        bQ.to_csv(out / "Q_summary_best.csv", index=False)
        fig_box(dfQ, bQ, out / "figures" / "Q_boxplot.png",
                "Benchmarks: FLOPTICS vs grid averaged across grid offsets")
        print("\n=== Experiment Q (best setting per method by mean ARI) ===")
        print(bQ[["dataset", "method", "setting", "ari_offset0", "ari_mean", "ari_std",
                  "ari_range", "k_min", "k_max"]].round(3).to_string(index=False))
        fl = bQ[bQ.method == "floptics"].set_index("dataset")
        av = bQ[bQ.method == "grid_averaged"].set_index("dataset")
        print(f"\nFLOPTICS datasets with ARI range > 0.10: {int((fl.ari_range > 0.10).sum())} / {len(fl)}")
        print(f"median ARI std: FLOPTICS = {fl.ari_std.median():.3f}  grid averaged = {av.ari_std.median():.3f}")
        print(f"median mean ARI: FLOPTICS = {fl.ari_mean.median():.3f}  grid averaged = {av.ari_mean.median():.3f}")
    print(f"\nTotal time {time.time() - t0:.1f}s. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
