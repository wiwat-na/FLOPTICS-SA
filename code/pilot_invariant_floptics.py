#!/usr/bin/env python3
"""
Offset-invariant FLOPTICS (working name: FLOPTICS-SA, "shift-averaged") - pilot.

Idea
  FLOPTICS = equal-sized bins + OPTICS on the non-empty bins. Its result changes when the
  grid is shifted (see pilot_floptics.py). FLOPTICS-SA keeps the two strengths of
  FLOPTICS (grid speed, no fixed epsilon) and removes the dependence on grid placement:

  1. Density that does not depend on where the grid starts:
     the bin counts are averaged over k shifts per axis (k^d grids, step h/k). This is the
     same as a fine grid (bin width h/k) smoothed with a triangle filter [ASH, Scott 1985].
  2. OPTICS-style density hierarchy instead of one fixed threshold:
     fine cells are visited from the densest to the sparsest (like OPTICS visiting points in
     order of reachability). Each new cell joins its already-visited neighbours
     (Chebyshev adjacency). When a cell connects two groups, the group with the lower peak
     is absorbed only if its peak rises less than `beta` above the connecting cell
     (its "persistence"). Groups that rise higher stay separate clusters.
     This merge rule is the persistence rule of ToMATo (Chazal et al., J. ACM 2013),
     here run directly on the grid.
  3. Cells with averaged density below `floor` are noise; clusters holding fewer than
     `min_frac` of the points are turned into noise.

  Densities and both thresholds are expressed in the same unit as the other pilots:
  multiples of the uniform mean N/m^d (points per coarse bin).

  Parameters: m (bins per axis, like Bin_size), k (shifts per axis), floor, beta.
  No epsilon and no MinPts.

Methods compared on the same random offsets
  floptics          re-implementation of the original (see pilot_floptics.py)
  floptics_sa       the new method, k shifts
  floptics_sa_k1    ablation: the same hierarchy on ONE grid (k = 1, no averaging)
  grid_averaged     the threshold version from pilot_grid_offset.py

Usage
  python pilot_invariant_floptics.py --quick
  python pilot_invariant_floptics.py                               # FLOPTICS paper sets A-D
  python pilot_invariant_floptics.py --data-dir clustering-data-v1 # plus benchmarks
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pilot_grid_offset import (DEFAULT_BENCHMARKS, grid_counts, load_benchmark,  # noqa: E402
                               normalise, parse_list, triangle_smooth)
from pilot_floptics import (floptics, grid_averaged, paper_dataset, scores)  # noqa: E402

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pure Python fallback: same result, slower
    HAVE_NUMBA = False

    def njit(*a, **k):
        def wrap(f):
            return f
        return wrap if not (len(a) == 1 and callable(a[0])) else a[0]


# ----------------------------------------------------------------------------- core
@njit(cache=True)
def _find(parent, i):
    root = i
    while parent[root] != root:
        root = parent[root]
    while parent[i] != root:  # path compression
        nxt = parent[i]
        parent[i] = root
        i = nxt
    return root


@njit(cache=True)
def _persistence_merge(flat, order, offsets, beta):
    """Visit cells from dense to sparse; union-find with the ToMATo persistence rule.
    flat: densities of a zero-padded grid (padding guarantees neighbours exist).
    Returns parent array (-1 = never visited)."""
    n = flat.size
    parent = np.full(n, -1, dtype=np.int64)
    peak = np.zeros(n)
    roots = np.empty(offsets.size, dtype=np.int64)
    for c in order:
        nr = 0
        for o in offsets:
            q = c + o
            if parent[q] != -1:
                r = _find(parent, q)
                dup = False
                for j in range(nr):
                    if roots[j] == r:
                        dup = True
                        break
                if not dup:
                    roots[nr] = r
                    nr += 1
        if nr == 0:
            parent[c] = c
            peak[c] = flat[c]
            continue
        best = roots[0]
        for j in range(1, nr):
            if peak[roots[j]] > peak[best]:
                best = roots[j]
        parent[c] = best
        for j in range(nr):
            r = roots[j]
            if r != best and peak[r] < flat[c] + beta:
                parent[r] = best  # younger group dies (elder rule)
    for i in range(n):  # final compression
        if parent[i] != -1:
            parent[i] = _find(parent, i)
    return parent


def chebyshev_offsets(shape):
    d = len(shape)
    strides = np.cumprod((1,) + tuple(shape[::-1]))[:-1][::-1]
    offs = []
    for delta in np.ndindex(*([3] * d)):
        delta = np.array(delta) - 1
        if np.any(delta != 0):
            offs.append(int(np.dot(delta, strides)))
    return np.array(offs, dtype=np.int64)


def floptics_sa(Xn, m, k, floor, beta, offset, min_frac=0.005):
    """FLOPTICS-SA. floor, beta in multiples of N/m^d. k=1 gives the single-grid ablation."""
    N, d = Xn.shape
    unit = N / m ** d
    c, f = grid_counts(Xn, m * k, offset * k, pad=k + 1)  # +1 so neighbour lookups stay inside
    D = triangle_smooth(c, k) if k > 1 else c
    D = D.copy()
    # keep a one-cell zero frame so no neighbour offset leaves the array
    for ax in range(d):
        sl = [slice(None)] * d
        sl[ax] = 0
        D[tuple(sl)] = 0
        sl[ax] = -1
        D[tuple(sl)] = 0
    flat = D.ravel()
    cand = np.flatnonzero(flat > floor * unit)
    order = cand[np.argsort(-flat[cand], kind="stable")]
    parent = _persistence_merge(flat, order, chebyshev_offsets(D.shape), beta * unit)
    root_of_point = parent[f]
    lab = np.full(N, -1)
    ok = root_of_point != -1
    _, lab_ok = np.unique(root_of_point[ok], return_inverse=True)
    lab[ok] = lab_ok
    # drop tiny clusters
    if ok.any():
        sizes = np.bincount(lab[ok])
        small = np.flatnonzero(sizes < max(2, min_frac * N))
        lab[np.isin(lab, small)] = -1
        keep = lab != -1
        _, lab[keep] = np.unique(lab[keep], return_inverse=True)
    return lab


# ----------------------------------------------------------------------------- experiment
def run(name, X, y, args, rng, settings, runs, include_zero=True):
    Xn = normalise(X)
    d = Xn.shape[1]
    offsets = rng.rand(runs, d)
    if include_zero:
        offsets[0] = 0.0
    rows, labels = [], {}
    for r, s in enumerate(offsets):
        for method, params in settings:
            t = time.perf_counter()
            if method == "floptics":
                lab = floptics(Xn, params["bins"], params["minpts"], s)
            elif method == "grid_averaged":
                lab = grid_averaged(Xn, params["m"], params["alpha"], args.k, s)
            else:
                k = args.k if method == "floptics_sa" else 1
                lab = floptics_sa(Xn, params["m"], k, params["floor"], params["beta"], s)
            sec = time.perf_counter() - t
            setting = ",".join(f"{a}={b}" for a, b in params.items())
            rows.append(dict(dataset=name, method=method, setting=setting, run=r,
                             offset=json.dumps(np.round(s, 4).tolist()), seconds=sec,
                             **scores(y, lab)))
            labels[(method, setting, r)] = lab
    return pd.DataFrame(rows), Xn, labels


def build_settings(args, paper):
    st = []
    if paper:
        st.append(("floptics", dict(bins=args.paper_bins, minpts=args.paper_minpts)))
    else:
        st += [("floptics", dict(bins=b, minpts=mp_)) for b in parse_list(args.q_bins, int)
               for mp_ in parse_list(args.q_minpts, int)]
    for m in parse_list(args.m, int):
        for fl in parse_list(args.floor, float):
            for be in parse_list(args.beta, float):
                p = dict(m=m, floor=fl, beta=be)
                st.append(("floptics_sa", p))
                st.append(("floptics_sa_k1", dict(p)))
        for a in parse_list(args.alpha, float):
            st.append(("grid_averaged", dict(m=m, alpha=a)))
    return st


def summarise(df):
    g = df.groupby(["dataset", "method", "setting"])
    s = g.agg(ari_mean=("ari", "mean"), ari_std=("ari", "std"), ari_min=("ari", "min"),
              ari_max=("ari", "max"), acc_mean=("paper_accuracy", "mean"),
              acc_std=("paper_accuracy", "std"), acc_min=("paper_accuracy", "min"),
              acc_max=("paper_accuracy", "max"), sec_mean=("seconds", "mean"),
              k_min=("n_clusters", "min"), k_max=("n_clusters", "max")).reset_index()
    s["ari_range"] = s.ari_max - s.ari_min
    return s


def best(summary):
    return summary.loc[summary.groupby(["dataset", "method"])["ari_mean"].idxmax()]


ORDER = ["floptics", "floptics_sa_k1", "grid_averaged", "floptics_sa"]
COL = {"floptics": "#d95f02", "floptics_sa_k1": "#7570b3", "grid_averaged": "#66a61e",
       "floptics_sa": "#1b9e77"}
LBL = {"floptics": "FLOPTICS (original)", "floptics_sa_k1": "hierarchy, single grid (k=1)",
       "grid_averaged": "averaged + fixed threshold", "floptics_sa": "FLOPTICS-SA (new)"}


def fig_box(df, b, path, title):
    names = list(dict.fromkeys(b.dataset))
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(names)), 4.8))
    data, pos, cols = [], [], []
    for i, nm in enumerate(names):
        for j, mt in enumerate(ORDER):
            bb = b[(b.dataset == nm) & (b.method == mt)]
            if bb.empty:
                continue
            v = df[(df.dataset == nm) & (df.method == mt) & (df.setting == bb.iloc[0].setting)].ari
            data.append(v.values)
            pos.append(i * (len(ORDER) + 1) + j)
            cols.append(COL[mt])
    bp = ax.boxplot(data, positions=pos, widths=0.8, patch_artist=True)
    for p_, c in zip(bp["boxes"], cols):
        p_.set_facecolor(c)
        p_.set_alpha(0.65)
    ax.set_xticks([i * (len(ORDER) + 1) + 1.5 for i in range(len(names))])
    ax.set_xticklabels([n.split("/")[-1] for n in names], rotation=45, ha="right")
    ax.set_ylabel("ARI across random grid offsets")
    ax.set_title(title)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COL[m], alpha=0.65) for m in ORDER]
    ax.legend(handles, [LBL[m] for m in ORDER], loc="lower left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_D(Xn, y, df, labels, b, path):
    """Dataset D: FLOPTICS at its worst offset vs FLOPTICS-SA at the same offset."""
    nm = "paper/D"
    bf = b[(b.dataset == nm) & (b.method == "floptics")].iloc[0]
    bs = b[(b.dataset == nm) & (b.method == "floptics_sa")].iloc[0]
    sub = df[(df.dataset == nm) & (df.method == "floptics") & (df.setting == bf.setting)]
    w = sub.loc[sub.ari.idxmin()]
    r = int(w.run)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, mt, st in ((axes[0], "floptics", bf.setting), (axes[1], "floptics_sa", bs.setting)):
        lab = labels[(mt, st, r)]
        sc = scores(y, lab)
        noise = lab == -1
        ax.scatter(Xn[noise, 0], Xn[noise, 1], s=2, c="lightgrey")
        ax.scatter(Xn[~noise, 0], Xn[~noise, 1], s=2, c=lab[~noise] % 10, cmap="tab10")
        ax.set_title(f"{LBL[mt]}\n{st}\nARI={sc['ari']:.3f}, accuracy={sc['paper_accuracy']:.1f}%",
                     fontsize=9)
        ax.set_aspect("equal")
    fig.suptitle(f"Dataset D, the same grid offset {json.loads(w.offset)} (FLOPTICS' worst)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--datasets", default=",".join(DEFAULT_BENCHMARKS))
    p.add_argument("--runs", type=int, default=30, help="offsets for the paper datasets")
    p.add_argument("--runs-q", type=int, default=20, help="offsets for the benchmarks")
    p.add_argument("--paper-bins", type=int, default=100)
    p.add_argument("--paper-minpts", type=int, default=10)
    p.add_argument("--q-bins", default="20,30,50")
    p.add_argument("--q-minpts", default="5,10")
    p.add_argument("--m", default="5,10,15,20,30", help="FLOPTICS-SA / grid averaged: bins per axis")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--floor", default="0.1,0.25,0.5", help="noise floor, multiples of N/m^d")
    p.add_argument("--beta", default="0.5,1,2,4", help="persistence, multiples of N/m^d")
    p.add_argument("--alpha", default="1,2,4", help="grid averaged: fixed threshold")
    p.add_argument("--max-n", type=int, default=6000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default="results_floptics_sa")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.runs, args.m, args.beta = 8, "20", "1"
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    print(f"numba: {'yes' if HAVE_NUMBA else 'no (pure Python, slower)'}")
    t0 = time.time()

    # warm-up compile so timings are fair
    floptics_sa(np.random.rand(50, 2), 5, 2, 0.5, 1.0, np.zeros(2))

    names = ["D"] if args.quick else ["A", "B", "C", "D"]
    settings = build_settings(args, paper=True)
    print(f"Paper datasets {names}: {len(settings)} settings x {args.runs} offsets")
    allP, keepD = [], None
    for nm in names:
        X, y = paper_dataset(nm, args.seed)
        t1 = time.time()
        df, Xn, labels = run(f"paper/{nm}", X, y, args, rng, settings, args.runs)
        allP.append(df)
        if nm == "D":
            keepD = (Xn, y, df, labels)
        print(f"  {nm}: {time.time() - t1:.1f}s")
    dfP = pd.concat(allP, ignore_index=True)
    dfP.to_csv(out / "P_runs.csv", index=False)
    sP = summarise(dfP)
    sP.to_csv(out / "P_summary_all.csv", index=False)
    bP = best(sP)
    bP.to_csv(out / "P_summary_best.csv", index=False)
    fig_box(dfP, bP, out / "figures" / "P_boxplot.png", "FLOPTICS paper datasets")
    if keepD is not None:
        fig_D(keepD[0], keepD[1], keepD[2], keepD[3], bP, out / "figures" / "P_D_same_offset.png")
    pd.set_option("display.width", 250)
    cols = ["dataset", "method", "setting", "acc_mean", "acc_std", "acc_min", "acc_max",
            "ari_mean", "ari_std", "ari_min", "k_min", "k_max", "sec_mean"]
    print("\n=== Paper datasets (accuracy % as in the FLOPTICS paper; best setting per method) ===")
    print(bP.sort_values(["dataset", "method"])[cols].round(3).to_string(index=False))

    if args.data_dir and not args.quick:
        settings_q = build_settings(args, paper=False)
        print(f"\nBenchmarks: {len(settings_q)} settings x {args.runs_q} offsets")
        allQ = []
        for nm in parse_list(args.datasets, str):
            res = load_benchmark(args.data_dir, nm)
            if res is None or res[0].shape[1] not in (2, 3) or len(res[1]) > args.max_n:
                continue
            t1 = time.time()
            df, _, _ = run(nm, res[0], res[1], args, rng, settings_q, args.runs_q)
            allQ.append(df)
            print(f"  {nm:22s} {time.time() - t1:6.1f}s")
        dfQ = pd.concat(allQ, ignore_index=True)
        dfQ.to_csv(out / "Q_runs.csv", index=False)
        sQ = summarise(dfQ)
        sQ.to_csv(out / "Q_summary_all.csv", index=False)
        bQ = best(sQ)
        bQ.to_csv(out / "Q_summary_best.csv", index=False)
        fig_box(dfQ, bQ, out / "figures" / "Q_boxplot.png", "Benchmarks")
        tab_m = bQ.pivot(index="dataset", columns="method", values="ari_mean")[ORDER]
        tab_s = bQ.pivot(index="dataset", columns="method", values="ari_std")[ORDER]
        print("\n=== Benchmarks: mean ARI over offsets (best setting per method) ===")
        print(tab_m.round(3).to_string())
        print("\n=== Benchmarks: ARI std over offsets ===")
        print(tab_s.round(3).to_string())
        print("\nmedians  mean ARI: " + "  ".join(f"{m}={tab_m[m].median():.3f}" for m in ORDER))
        print("medians  ARI std : " + "  ".join(f"{m}={tab_s[m].median():.3f}" for m in ORDER))
        for other in ("floptics", "floptics_sa_k1", "grid_averaged"):
            dlt = tab_m["floptics_sa"] - tab_m[other]
            print(f"FLOPTICS-SA vs {other:15s}: higher on {int((dlt > 0.01).sum())}, "
                  f"lower on {int((dlt < -0.01).sum())}, tie on {int((dlt.abs() <= 0.01).sum())}")
    print(f"\nTotal time {time.time() - t0:.1f}s. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
