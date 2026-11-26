#!/usr/bin/env python3
"""
Pilot study: How sensitive is grid-based density clustering to the grid offset,
and does averaging over shifted grids (ASH-style) reduce that sensitivity?

Three methods are compared on the SAME random offsets (paired design):

  single   : CLIQUE-style full-space grid clustering.
             Each axis is cut into m equal bins, shifted by a random offset s (fraction of a bin).
             A cell is dense if count > alpha * N / m^d  (relative threshold, see note on tau).
             Dense cells that touch are merged (connected components).

  averaged : Density averaged over k shifts per axis (k^d grids, shift step h/k).
             This equals a fine grid (bin width h/k) whose counts are smoothed with a
             separable triangle filter of weights (k-|j|)/k  [Averaged Shifted Histogram,
             Scott 1985]. Same threshold rule, connected components on the fine grid.
             NOTE: this averages the DENSITY, not the final cluster decisions.

  single_fine : CONTROL. One grid at the fine resolution m*k (no averaging), threshold
             expressed as the same density. If "averaged" beats "single", this tells
             whether the gain comes from averaging or merely from the finer resolution.

Baselines in experiment A (--no-baselines to skip): DBSCAN and HDBSCAN (scikit-learn),
tuned over a small parameter grid with the same best-by-ARI rule.

Experiment C (--no-scaling to skip): runtime and ARI as N grows from 10^4 to 10^6
(2D and 3D labelled mixtures). Parameters are tuned at the smallest N, then scaled so each
method keeps a density threshold proportional to N. Each run happens in its own process
and is stopped after --time-budget seconds.

Experiment B (on by default, --no-boundary to skip): synthetic 2D shapes with KNOWN boundaries
(rotated square, disk, annulus) plus uniform background noise. Measures
  iou        : area overlap between the predicted dense region and the true shapes
  band_error : in/out misclassification rate of points within one coarse cell of a
               true boundary (this is where staircase errors live)

Adjacency:
  face       : cells must share a face (original CLIQUE; 4-neighbour in 2D)
  chebyshev  : cells may touch at a corner (8-neighbour in 2D; used in the Symmetry paper)

Metrics per run:
  ari, nmi       : agreement with the reference labels (predicted noise = one extra label)
  accuracy       : share of points labelled correctly after matching found clusters to
                   reference clusters one-to-one (Hungarian method); noise counts as wrong
  acc_boundary   : the same accuracy, but only on "boundary points" (points whose 10
                   nearest neighbours contain a different reference label)
  noise_frac     : fraction of points left as noise
  n_clusters     : number of clusters found
Stability (no ground truth needed) is computed afterwards as ARI between consecutive runs.

Usage (see the Thai step-by-step guide that came with this script):
  python pilot_grid_offset.py --quick                       # synthetic data only, ~1 minute
  python pilot_grid_offset.py --data-dir clustering-data-v1 # full pilot with benchmark data
"""

import argparse
import gzip
import json
import multiprocessing as mp
import queue
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=FutureWarning)
import matplotlib.pyplot as plt  # noqa: E402

# 2D and 3D datasets from the Gagolewski benchmark suite (clustering-data-v1)
DEFAULT_BENCHMARKS = [
    "sipu/aggregation", "sipu/compound", "sipu/jain", "sipu/flame",
    "sipu/pathbased", "sipu/spiral", "sipu/r15", "sipu/d31", "sipu/s1",
    "fcps/lsun", "fcps/target", "wut/smile", "other/hdbscan",
    "fcps/atom", "fcps/chainlink", "fcps/hepta", "fcps/tetra",
]


# ----------------------------------------------------------------------------- data
def synthetic_datasets(seed):
    rng = np.random.RandomState(seed)
    out = {}
    X, y = make_moons(n_samples=1000, noise=0.06, random_state=rng)
    out["synthetic/moons"] = (X, y)
    X, y = make_circles(n_samples=1000, factor=0.5, noise=0.05, random_state=rng)
    out["synthetic/circles"] = (X, y)
    X, y = make_blobs(n_samples=1500, centers=[[0, 0], [6, 0], [3, 5]],
                      cluster_std=[1.0, 1.8, 0.5], random_state=rng)
    out["synthetic/blobs_varied"] = (X, y)
    X, y = make_blobs(n_samples=1500, n_features=3, centers=4, cluster_std=0.8,
                      random_state=rng)
    out["synthetic/blobs3d"] = (X, y)
    return out


def load_benchmark(data_dir, name):
    base = Path(data_dir) / name
    data_file = base.with_suffix(".data.gz")
    lab_file = Path(str(base) + ".labels0.gz")
    if not data_file.exists() or not lab_file.exists():
        return None
    with gzip.open(data_file, "rt") as f:
        X = np.loadtxt(f, ndmin=2)
    with gzip.open(lab_file, "rt") as f:
        y = np.loadtxt(f, dtype=int)
    return X, y


def load_csv(path):
    """CSV/TXT: numeric feature columns, LAST column = label (0 or -1 = noise allowed)."""
    df = pd.read_csv(path, sep=None, engine="python")
    X = df.iloc[:, :-1].to_numpy(dtype=float)
    y = pd.factorize(df.iloc[:, -1])[0]
    return X, y


def normalise(X):
    """Min-max scale every axis to [0, 1) so that m bins per axis cover the data."""
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    return (X - lo) / span * (1 - 1e-9)


# ----------------------------------------------------------------------------- clustering
def structure(d, adjacency):
    return ndimage.generate_binary_structure(d, 1 if adjacency == "face" else d)


def grid_counts(Xn, nbins, offset_cells, pad):
    """Histogram on a regular grid. offset_cells = shift in units of this grid's cell."""
    d = Xn.shape[1]
    idx = np.floor(Xn * nbins + offset_cells).astype(np.int64) + pad
    extra = int(np.ceil(np.max(offset_cells))) + 1  # offset may exceed one cell (fine grids)
    shape = tuple([nbins + extra + 2 * pad] * d)
    flat = np.ravel_multi_index(idx.T, shape)
    counts = np.bincount(flat, minlength=int(np.prod(shape))).reshape(shape).astype(float)
    return counts, flat


def triangle_smooth(counts, k):
    w = (k - np.abs(np.arange(-(k - 1), k))) / k  # weights (k-|j|)/k
    out = counts
    for ax in range(counts.ndim):
        out = ndimage.correlate1d(out, w, axis=ax, mode="constant", cval=0.0)
    return out


def label_points(density, flat_idx, threshold, struct):
    dense = density > threshold
    comp, ncomp = ndimage.label(dense, structure=struct)
    lab = comp.ravel()[flat_idx].astype(np.int64)
    lab[lab == 0] = -1  # noise
    return lab, ncomp


# ----------------------------------------------------------------------------- metrics
def boundary_mask(Xn, y, k=10):
    k = min(k, len(y) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xn)
    _, ind = nn.kneighbors(Xn)
    return (y[ind[:, 1:]] != y[:, None]).any(axis=1)


def matched_correct(y, lab):
    """Map found clusters to reference labels one-to-one (Hungarian method) and
    return a boolean array: is each point's label correct? Predicted noise counts as wrong."""
    yc = pd.factorize(y)[0]
    clusters = np.unique(lab[lab != -1])
    correct = np.zeros(len(y), dtype=bool)
    if len(clusters) == 0:
        return correct
    lc = np.searchsorted(clusters, lab[lab != -1])
    cont = np.zeros((len(clusters), yc.max() + 1))
    np.add.at(cont, (lc, yc[lab != -1]), 1)
    rows, cols = linear_sum_assignment(-cont)
    mapping = np.full(len(clusters), -2)
    mapping[rows] = cols
    pred = np.full(len(y), -1)
    pred[lab != -1] = mapping[lc]
    return pred == yc


def evaluate(y, lab, bmask):
    ari = adjusted_rand_score(y, lab)
    nmi = normalized_mutual_info_score(y, lab)
    correct = matched_correct(y, lab)
    acc = float(correct.mean())
    acc_b = float(correct[bmask].mean()) if bmask.sum() >= 10 else np.nan
    n_clu = len(set(lab.tolist()) - {-1})
    return ari, nmi, acc, acc_b, float((lab == -1).mean()), n_clu


# ----------------------------------------------------------------------------- experiment
def run_dataset(name, X, y, args, rng):
    Xn = normalise(X)
    N, d = Xn.shape
    bmask = boundary_mask(Xn, y)
    offsets = rng.rand(args.runs, d)  # shared by both methods (paired design)
    rows, labels_store = [], {}
    for m in args.m:
        for r, s in enumerate(offsets):
            # single grid: m bins, shift s (in bins)
            c_single, f_single = grid_counts(Xn, m, s, pad=0)
            # averaged: fine grid m*k bins, same physical shift s*h = s*k fine cells
            k = args.k
            c_fine, f_fine = grid_counts(Xn, m * k, s * k, pad=k)
            c_avg = triangle_smooth(c_fine, k)
            # control: ONE grid at the fine resolution m*k, no averaging
            c_sf, f_sf = grid_counts(Xn, m * k, s * k, pad=0)
            for alpha in args.alpha:
                thr = alpha * N / (m ** d)              # coarse-cell units
                thr_fine = alpha * N / ((m * k) ** d)   # same density, fine-cell units
                for adj in args.adjacency:
                    st = structure(d, adj)
                    for method, dens, flat, th in (("single", c_single, f_single, thr),
                                                   ("averaged", c_avg, f_fine, thr),
                                                   ("single_fine", c_sf, f_sf, thr_fine)):
                        lab, _ = label_points(dens, flat, th, st)
                        ari, nmi, acc, acc_b, nf, nc = evaluate(y, lab, bmask)
                        rows.append(dict(dataset=name, n=N, d=d, m=m, alpha=alpha,
                                         adjacency=adj, method=method, run=r,
                                         offset=json.dumps(np.round(s, 4).tolist()),
                                         ari=ari, nmi=nmi, accuracy=acc,
                                         acc_boundary=acc_b,
                                         noise_frac=nf, n_clusters=nc))
                        labels_store[(m, alpha, adj, method, r)] = lab
    return pd.DataFrame(rows), Xn, offsets, labels_store


def stability(df_runs, labels_store):
    """ARI between consecutive runs (different offsets), no ground truth needed."""
    out = []
    keys = df_runs.groupby(["m", "alpha", "adjacency", "method"]).size().index
    for (m, alpha, adj, method) in keys:
        runs = sorted(r for (mm, aa, jj, me, r) in labels_store
                      if (mm, aa, jj, me) == (m, alpha, adj, method))
        vals = [adjusted_rand_score(labels_store[(m, alpha, adj, method, a)],
                                    labels_store[(m, alpha, adj, method, b)])
                for a, b in zip(runs[:-1], runs[1:])]
        out.append(dict(m=m, alpha=alpha, adjacency=adj, method=method,
                        stability_ari=float(np.mean(vals)) if vals else np.nan))
    return pd.DataFrame(out)


def summarise(df):
    g = df.groupby(["dataset", "n", "d", "m", "alpha", "adjacency", "method"])
    s = g.agg(ari_mean=("ari", "mean"), ari_std=("ari", "std"),
              ari_min=("ari", "min"), ari_max=("ari", "max"),
              nmi_mean=("nmi", "mean"), acc_mean=("accuracy", "mean"),
              acc_boundary_mean=("acc_boundary", "mean"),
              noise_frac_mean=("noise_frac", "mean"),
              n_clusters_min=("n_clusters", "min"), n_clusters_max=("n_clusters", "max"))
    s["ari_range"] = s["ari_max"] - s["ari_min"]
    return s.reset_index()


def best_settings(summary, adjacency):
    """For each dataset and method, keep the (m, alpha) with the highest mean ARI.
    Optimistic (tuned on the reference labels) but applied identically to all methods."""
    sub = summary[summary.adjacency == adjacency]
    idx = sub.groupby(["dataset", "method"])["ari_mean"].idxmax()
    return sub.loc[idx].sort_values(["dataset", "method"])



METHODS = ["single", "single_fine", "averaged"]
COLORS = {"single": "#d95f02", "single_fine": "#7570b3", "averaged": "#1b9e77"}
NAMES = {"single": "single grid (m)", "single_fine": "single grid (m*k), control",
         "averaged": "averaged (k shifts)"}


# ----------------------------------------------------------------------------- figures
def fig_boxplot(df, best, adjacency, path):
    names = sorted(best.dataset.unique())
    nm_ = len(METHODS)
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(names)), 4.5))
    pos, data, colors = [], [], []
    for i, nm in enumerate(names):
        for j, method in enumerate(METHODS):
            b = best[(best.dataset == nm) & (best.method == method)].iloc[0]
            v = df[(df.dataset == nm) & (df.method == method) & (df.m == b.m)
                   & (df.alpha == b.alpha) & (df.adjacency == adjacency)].ari
            data.append(v.values)
            pos.append(i * (nm_ + 1) + j)
            colors.append(COLORS[method])
    bp = ax.boxplot(data, positions=pos, widths=0.8, patch_artist=True, showfliers=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_xticks([i * (nm_ + 1) + (nm_ - 1) / 2 for i in range(len(names))])
    ax.set_xticklabels([n.split("/")[-1] for n in names], rotation=45, ha="right")
    ax.set_ylabel("ARI across random grid offsets")
    ax.set_title(f"Offset sensitivity (best m, alpha per method; adjacency={adjacency})")
    ax.legend([bp["boxes"][j] for j in range(nm_)], [NAMES[m] for m in METHODS],
              loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_example(name, Xn, y, df, labels_store, best, adjacency, path):
    """Single grid at its worst and best offset, then the control and the averaged
    method at that same worst offset (each method with its own best m, alpha)."""
    b = best[(best.dataset == name) & (best.method == "single")].iloc[0]
    sub = df[(df.dataset == name) & (df.method == "single") & (df.m == b.m)
             & (df.alpha == b.alpha) & (df.adjacency == adjacency)]
    worst, bestr = sub.loc[sub.ari.idxmin()], sub.loc[sub.ari.idxmax()]
    m = int(b.m)
    panels = [("single, worst offset", worst, "single"),
              ("single, best offset", bestr, "single"),
              ("single_fine (control), same worst offset", worst, "single_fine"),
              ("averaged, same worst offset", worst, "averaged")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (title, row, method) in zip(axes, panels):
        bm = best[(best.dataset == name) & (best.method == method)].iloc[0]
        lab = labels_store[(int(bm.m), bm.alpha, adjacency, method, int(row.run))]
        ari = adjusted_rand_score(y, lab)
        noise = lab == -1
        ax.scatter(Xn[noise, 0], Xn[noise, 1], s=4, c="lightgrey")
        ax.scatter(Xn[~noise, 0], Xn[~noise, 1], s=6, c=lab[~noise] % 20, cmap="tab20")
        if method == "single":
            s = json.loads(row.offset)
            for t in range(-1, m + 2):
                ax.axvline((t - s[0]) / m, lw=0.3, c="k", alpha=0.3)
                ax.axhline((t - s[1]) / m, lw=0.3, c="k", alpha=0.3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_title(f"{title}\nm={int(bm.m)}, alpha={bm.alpha}, ARI = {ari:.3f}", fontsize=10)
    fig.suptitle(f"{name}  (adjacency={adjacency})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_sweep(name, Xn, y, m, alpha, k, adjacency, path, steps=60):
    """ARI as the grid slides diagonally by one full bin (offset s = (t, t, ...)).
    All three methods use the same m and alpha here, so the curves are directly comparable."""
    N, d = Xn.shape
    st = structure(d, adjacency)
    thr, thr_fine = alpha * N / (m ** d), alpha * N / ((m * k) ** d)
    ts = np.linspace(0, 1, steps, endpoint=False)
    res = {mt: [] for mt in METHODS}
    for t in ts:
        s = np.full(d, t)
        c, f = grid_counts(Xn, m, s, pad=0)
        res["single"].append(adjusted_rand_score(y, label_points(c, f, thr, st)[0]))
        c, f = grid_counts(Xn, m * k, s * k, pad=0)
        res["single_fine"].append(adjusted_rand_score(y, label_points(c, f, thr_fine, st)[0]))
        c, f = grid_counts(Xn, m * k, s * k, pad=k)
        res["averaged"].append(
            adjusted_rand_score(y, label_points(triangle_smooth(c, k), f, thr, st)[0]))
    fig, ax = plt.subplots(figsize=(7, 4))
    for mt in METHODS:
        ax.plot(ts, res[mt], "-o", ms=3, c=COLORS[mt], label=NAMES[mt])
    ax.set_xlabel("grid offset (fraction of one coarse bin, along the diagonal)")
    ax.set_ylabel("ARI")
    ax.set_title(f"{name}: ARI while sliding the grid  (m={m}, k={k}, alpha={alpha})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- experiment B
def shapes_sdf(P):
    """Signed distance (negative = inside) to three known shapes in [0,1]^2.
    Returns (sdf_union, shape_id) where shape_id is 1..3 inside a shape and 0 outside."""
    # 1) square, side 0.30, rotated 30 degrees, centre (0.28, 0.72)
    c, a, th = np.array([0.28, 0.72]), 0.15, np.deg2rad(30)
    R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    q = np.abs((P - c) @ R.T) - a
    sq = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
    # 2) disk, radius 0.15, centre (0.75, 0.72)
    dk = np.linalg.norm(P - [0.75, 0.72], axis=1) - 0.15
    # 3) annulus, radii 0.11 to 0.21, centre (0.50, 0.27)
    r = np.linalg.norm(P - [0.50, 0.27], axis=1)
    an = np.maximum(0.11 - r, r - 0.21)
    all_ = np.stack([sq, dk, an], axis=1)
    sdf = all_.min(axis=1)
    sid = np.where(sdf < 0, all_.argmin(axis=1) + 1, 0)
    return sdf, sid


def make_shapes(n_in, n_noise, rng):
    pts = []
    while sum(len(p) for p in pts) < n_in:
        cand = rng.rand(20000, 2)
        pts.append(cand[shapes_sdf(cand)[0] < 0])
    inside = np.concatenate(pts)[:n_in]
    X = np.vstack([inside, rng.rand(n_noise, 2)])
    sdf, sid = shapes_sdf(X)
    return X, sdf, sid


def boundary_experiment(args, rng, out):
    """Known-boundary test for the staircase effect (2D only).
    The data already live in [0,1)^2, so no rescaling is applied."""
    X, sdf, sid = make_shapes(args.shape_n, args.shape_noise, rng)
    N, d = X.shape
    inside_true = sid > 0
    g = (np.arange(args.probe) + 0.5) / args.probe
    P = np.stack(np.meshgrid(g, g, indexing="xy"), axis=-1).reshape(-1, 2)
    region_true = shapes_sdf(P)[0] < 0
    k = args.k
    offsets = rng.rand(args.runs, d)
    rows, keep = [], {}
    for m in args.m:
        band = np.abs(sdf) < 1.0 / m      # points within one coarse cell of a true boundary
        for r, s in enumerate(offsets):
            dens = {}
            c, f = grid_counts(X, m, s, pad=0)
            fp = np.ravel_multi_index(
                np.floor(P * m + s).astype(np.int64).T, c.shape)
            dens["single"] = (c, f, fp, 1.0)
            c, f = grid_counts(X, m * k, s * k, pad=0)
            fp = np.ravel_multi_index(
                np.floor(P * m * k + s * k).astype(np.int64).T, c.shape)
            dens["single_fine"] = (c, f, fp, 1.0 / k ** d)
            cf, f = grid_counts(X, m * k, s * k, pad=k)
            fp = np.ravel_multi_index(
                (np.floor(P * m * k + s * k).astype(np.int64) + k).T, cf.shape)
            dens["averaged"] = (triangle_smooth(cf, k), f, fp, 1.0)
            for alpha in args.alpha:
                thr = alpha * N / (m ** d)
                for method, (D, fpts, fprobe, scale) in dens.items():
                    dense = (D > thr * scale).ravel()
                    pred_in = dense[fpts]
                    region = dense[fprobe]
                    inter = (region & region_true).sum()
                    union = (region | region_true).sum()
                    rows.append(dict(m=m, alpha=alpha, method=method, run=r,
                                     iou=inter / union,
                                     band_error=float((pred_in[band] != inside_true[band]).mean()),
                                     overall_error=float((pred_in != inside_true).mean())))
                    if r == 0:
                        keep[(m, alpha, method)] = region.reshape(args.probe, args.probe)
    df = pd.DataFrame(rows)
    df.to_csv(out / "boundary_runs.csv", index=False)
    summ = (df.groupby(["m", "alpha", "method"])
              .agg(iou_mean=("iou", "mean"), iou_std=("iou", "std"),
                   band_error_mean=("band_error", "mean"),
                   band_error_std=("band_error", "std"),
                   overall_error_mean=("overall_error", "mean"))
              .reset_index())
    summ.to_csv(out / "boundary_summary.csv", index=False)

    # figure: true outline vs predicted region at run 0, for one representative (m, alpha)
    m_show = args.m[len(args.m) // 2]
    a_show = min(args.alpha, key=lambda a: abs(a - 2.0))
    gt = shapes_sdf(P)[0].reshape(args.probe, args.probe)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    for ax, method in zip(axes, METHODS):
        reg = keep[(m_show, a_show, method)]
        ax.imshow(reg, origin="lower", extent=(0, 1, 0, 1), cmap="Greens", alpha=0.6,
                  vmin=0, vmax=1.4)
        ax.contour(g, g, gt, levels=[0], colors="red", linewidths=1.2)
        ax.scatter(X[:, 0], X[:, 1], s=0.5, c="k", alpha=0.3)
        row = summ[(summ.m == m_show) & (summ.alpha == a_show) & (summ.method == method)].iloc[0]
        ax.set_title(f"{NAMES[method]}\nIoU={row.iou_mean:.3f}  band error={row.band_error_mean:.3f}"
                     f"  (mean of {args.runs} offsets)", fontsize=10)
        ax.set_aspect("equal")
    fig.suptitle(f"Known boundaries (red) vs predicted dense region (green), m={m_show}, "
                 f"alpha={a_show}, k={k}, offset of run 0")
    fig.tight_layout()
    fig.savefig(out / "figures" / "boundary_shapes.png", dpi=150)
    plt.close(fig)
    return summ


# ----------------------------------------------------------------------------- baselines (A)
DBSCAN_EPS = [0.01, 0.02, 0.03, 0.05, 0.08]
DBSCAN_MINPTS = [5, 10, 20]
HDBSCAN_MCS = [5, 10, 20, 40]


def run_baselines(name, Xn, y):
    """DBSCAN and HDBSCAN on the same rescaled data. Both are deterministic and do not
    depend on any grid, so there is no offset to vary: one run per setting.
    The best setting is chosen by ARI against the reference labels, exactly like the
    grid methods (optimistic for every method alike)."""
    rows = []
    for eps in DBSCAN_EPS:
        for ms in DBSCAN_MINPTS:
            t = time.perf_counter()
            lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(Xn)
            rows.append(dict(dataset=name, method="dbscan", params=f"eps={eps},min_samples={ms}",
                             seconds=time.perf_counter() - t,
                             ari=adjusted_rand_score(y, lab),
                             nmi=normalized_mutual_info_score(y, lab),
                             noise_frac=float((lab == -1).mean()),
                             n_clusters=len(set(lab.tolist()) - {-1})))
    for mcs in HDBSCAN_MCS:
        t = time.perf_counter()
        lab = HDBSCAN(min_cluster_size=mcs).fit_predict(Xn)
        rows.append(dict(dataset=name, method="hdbscan", params=f"min_cluster_size={mcs}",
                         seconds=time.perf_counter() - t,
                         ari=adjusted_rand_score(y, lab),
                         nmi=normalized_mutual_info_score(y, lab),
                         noise_frac=float((lab == -1).mean()),
                         n_clusters=len(set(lab.tolist()) - {-1})))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- experiment C
def make_scaling_data(n, d, seed):
    """Labelled mixture at any size: 2D = two moons + three blobs of different spread,
    3D = five blobs; plus 5% uniform background noise (label 0). Rescaled to [0,1)^d."""
    rng = np.random.RandomState(seed)
    n_noise = int(0.05 * n)
    n_sig = n - n_noise
    if d == 2:
        n_moon = int(0.4 * n_sig)
        Xm, ym = make_moons(n_samples=n_moon, noise=0.05, random_state=rng)
        Xm = Xm * 0.22 + [0.30, 0.62]
        Xb, yb = make_blobs(n_samples=n_sig - n_moon,
                            centers=[[0.75, 0.75], [0.25, 0.22], [0.72, 0.25]],
                            cluster_std=[0.035, 0.06, 0.025], random_state=rng)
        X = np.vstack([Xm, Xb])
        y = np.concatenate([ym + 1, yb + 3])
    else:
        centres = np.random.RandomState(7).rand(5, 3) * 0.6 + 0.2  # fixed for every n
        X, y = make_blobs(n_samples=n_sig, n_features=3, centers=centres,
                          cluster_std=[0.03, 0.04, 0.05, 0.03, 0.06], random_state=rng)
        y = y + 1
    X = np.clip(X, 0, 1 - 1e-9)
    X = np.vstack([X, rng.rand(n_noise, d) * (1 - 1e-9)])
    y = np.concatenate([y, np.zeros(n_noise, dtype=int)])
    return X, y


def _cluster_once(method, X, params):
    """Run one method once; returns labels. Used inside a child process."""
    N, d = X.shape
    if method == "grid_single":
        m, alpha, adj = params["m"], params["alpha"], params["adjacency"]
        c, f = grid_counts(X, m, np.zeros(d), pad=0)
        return label_points(c, f, alpha * N / m ** d, structure(d, adj))[0]
    if method == "grid_averaged":
        m, alpha, k, adj = params["m"], params["alpha"], params["k"], params["adjacency"]
        c, f = grid_counts(X, m * k, np.zeros(d), pad=k)
        return label_points(triangle_smooth(c, k), f, alpha * N / m ** d, structure(d, adj))[0]
    if method == "dbscan":
        return DBSCAN(eps=params["eps"], min_samples=params["min_samples"]).fit_predict(X)
    if method == "hdbscan":
        return HDBSCAN(min_cluster_size=params["min_cluster_size"],
                       min_samples=params["min_samples"]).fit_predict(X)
    raise ValueError(method)


def _child(method, X, params, q):
    t = time.perf_counter()
    lab = _cluster_once(method, X, params)
    q.put((time.perf_counter() - t, lab))


def timed_run(method, X, params, timeout):
    """Run in a separate process so a slow method can be stopped at `timeout` seconds."""
    q = mp.Queue()
    p = mp.Process(target=_child, args=(method, X, params, q))
    p.start()
    try:
        secs, lab = q.get(timeout=timeout)
    except queue.Empty:
        p.terminate()
        p.join()
        return None, None
    p.join()
    return secs, lab


def scaling_params(n, n0, d, tuned, args):
    """Keep every method's density threshold proportional to N (same rule as alpha*N/m^d):
    grid: m fixed, threshold grows with N automatically;
    DBSCAN: min_samples fixed, eps shrinks as (n0/n)^(1/d);
    HDBSCAN: min_samples fixed, min_cluster_size grows with n."""
    r = n / n0
    return {
        "grid_single": dict(m=tuned["grid_single"]["m"], alpha=tuned["grid_single"]["alpha"],
                            adjacency=args.scale_adjacency),
        "grid_averaged": dict(m=tuned["grid_averaged"]["m"],
                              alpha=tuned["grid_averaged"]["alpha"],
                              k=args.k, adjacency=args.scale_adjacency),
        "dbscan": dict(eps=tuned["dbscan"]["eps"] * r ** (-1.0 / d),
                       min_samples=tuned["dbscan"]["min_samples"]),
        "hdbscan": dict(min_cluster_size=max(5, int(round(tuned["hdbscan"]["min_cluster_size"] * r))),
                        min_samples=10),
    }


def tune_at_n0(X, y, args):
    """Pick each method's parameters on the smallest size by ARI (same oracle rule)."""
    N, d = X.shape
    best = {}
    cands = {
        "grid_single": [dict(m=m, alpha=a, adjacency=args.scale_adjacency)
                        for m in args.m for a in args.alpha],
        "grid_averaged": [dict(m=m, alpha=a, k=args.k, adjacency=args.scale_adjacency)
                          for m in args.m for a in args.alpha],
        "dbscan": [dict(eps=e, min_samples=s) for e in DBSCAN_EPS for s in DBSCAN_MINPTS],
        "hdbscan": [dict(min_cluster_size=c, min_samples=10) for c in [20, 50, 100, 200]],
    }
    for method, plist in cands.items():
        scores = [adjusted_rand_score(y, _cluster_once(method, X, p)) for p in plist]
        best[method] = plist[int(np.argmax(scores))]
    return best


def scaling_experiment(args, out):
    sizes = parse_list(args.scale_sizes, int)
    rows = []
    for d in (2, 3):
        X0, y0 = make_scaling_data(sizes[0], d, args.seed)
        tuned = tune_at_n0(X0, y0, args)
        print(f"  d={d} parameters tuned at n={sizes[0]}: "
              + "; ".join(f"{k}={v}" for k, v in tuned.items()))
        stopped = set()
        for n in sizes:
            X, y = make_scaling_data(n, d, args.seed + n)
            params = scaling_params(n, sizes[0], d, tuned, args)
            for method in METHODS_C:
                if method in stopped:
                    rows.append(dict(d=d, n=n, method=method, seconds=np.nan, ari=np.nan,
                                     status="skipped (slower size timed out)"))
                    continue
                reps = 3 if n <= 100_000 else 1
                times, lab = [], None
                for _ in range(reps):
                    secs, lab = timed_run(method, X, params[method], args.time_budget)
                    if secs is None:
                        break
                    times.append(secs)
                if not times:
                    stopped.add(method)
                    rows.append(dict(d=d, n=n, method=method, seconds=np.nan, ari=np.nan,
                                     status=f"timeout > {args.time_budget}s"))
                    print(f"    d={d} n={n:>9,} {method:14s} TIMEOUT")
                    continue
                ari = adjusted_rand_score(y, lab)
                rows.append(dict(d=d, n=n, method=method, seconds=float(np.median(times)),
                                 ari=ari, status="ok", params=json.dumps(params[method])))
                print(f"    d={d} n={n:>9,} {method:14s} {np.median(times):9.3f}s  ARI={ari:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "scaling.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    col = {"grid_single": "#d95f02", "grid_averaged": "#1b9e77",
           "dbscan": "#7570b3", "hdbscan": "#e7298a"}
    for ax, d in zip(axes, (2, 3)):
        for method in METHODS_C:
            s = df[(df.d == d) & (df.method == method) & (df.status == "ok")]
            ax.loglog(s.n, s.seconds, "-o", c=col[method], label=method)
        ax.set_xlabel("number of points N")
        ax.set_ylabel("seconds (median)")
        ax.set_title(f"Runtime, d={d} (timeout {args.time_budget}s)")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures" / "scaling_runtime.png", dpi=150)
    plt.close(fig)
    return df


METHODS_C = ["grid_single", "grid_averaged", "dbscan", "hdbscan"]


# ----------------------------------------------------------------------------- main
def parse_list(s, typ):
    return [typ(v) for v in s.split(",") if v.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None,
                   help="folder of clustering-data-v1 (contains sipu/, fcps/, wut/, other/)")
    p.add_argument("--datasets", default=",".join(DEFAULT_BENCHMARKS),
                   help="comma-separated benchmark names, e.g. sipu/jain,fcps/atom")
    p.add_argument("--csv", nargs="*", default=[],
                   help="extra CSV files: feature columns..., label column last")
    p.add_argument("--no-synthetic", action="store_true", help="skip the built-in synthetic sets")
    p.add_argument("--no-boundary", action="store_true", help="skip experiment B (known shapes)")
    p.add_argument("--shape-n", type=int, default=6000, help="points inside the known shapes")
    p.add_argument("--shape-noise", type=int, default=1000, help="uniform background points")
    p.add_argument("--probe", type=int, default=400, help="probe lattice size for IoU")
    p.add_argument("--m", default="10,15,20,30", help="bins per axis to try")
    p.add_argument("--alpha", default="1,2,4",
                   help="density threshold as a multiple of the uniform mean N/m^d")
    p.add_argument("--k", type=int, default=4, help="shifts per axis for the averaged method")
    p.add_argument("--adjacency", default="face,chebyshev")
    p.add_argument("--runs", type=int, default=100, help="random offsets per setting")
    p.add_argument("--max-n", type=int, default=6000,
                   help="skip datasets larger than this (keeps the pilot fast)")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default="results")
    p.add_argument("--no-baselines", action="store_true",
                   help="skip DBSCAN/HDBSCAN in experiment A")
    p.add_argument("--no-scaling", action="store_true", help="skip experiment C (runtime)")
    p.add_argument("--scale-sizes", default="10000,30000,100000,300000,1000000",
                   help="dataset sizes for experiment C")
    p.add_argument("--scale-adjacency", default="chebyshev")
    p.add_argument("--time-budget", type=int, default=300,
                   help="seconds before a method is stopped in experiment C")
    p.add_argument("--quick", action="store_true",
                   help="smoke test: synthetic data only, 20 runs, m=10,20, alpha=2")
    args = p.parse_args()

    if args.quick:
        args.runs, args.m, args.alpha, args.data_dir = 20, "10,20", "2", None
        args.scale_sizes, args.time_budget = "5000,20000", 60
    args.m = parse_list(args.m, int)
    args.alpha = parse_list(args.alpha, float)
    args.adjacency = parse_list(args.adjacency, str)
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    # ---- collect datasets
    datasets = {}
    if not args.no_synthetic:
        datasets.update(synthetic_datasets(args.seed))
    if args.data_dir:
        for nm in parse_list(args.datasets, str):
            res = load_benchmark(args.data_dir, nm)
            if res is None:
                print(f"[skip] {nm}: files not found under {args.data_dir}")
            else:
                datasets[nm] = res
    for path in args.csv:
        datasets["csv/" + Path(path).stem] = load_csv(path)

    print(f"Settings: m={args.m} alpha={args.alpha} k={args.k} "
          f"adjacency={args.adjacency} runs={args.runs}")
    print("Experiment A: offset sensitivity on labelled datasets")
    all_runs, all_stab, all_base = [], [], []
    store = {}
    t0 = time.time()
    for nm, (X, y) in datasets.items():
        if X.shape[1] not in (2, 3):
            print(f"[skip] {nm}: d={X.shape[1]} (pilot covers 2D/3D only)")
            continue
        if len(y) > args.max_n:
            print(f"[skip] {nm}: n={len(y)} > --max-n {args.max_n}")
            continue
        t1 = time.time()
        df, Xn, _, labels = run_dataset(nm, X, np.asarray(y), args, rng)
        stab = stability(df, labels)
        stab.insert(0, "dataset", nm)
        all_runs.append(df)
        all_stab.append(stab)
        store[nm] = (Xn, np.asarray(y), labels)
        if not args.no_baselines:
            all_base.append(run_baselines(nm, Xn, np.asarray(y)))
        print(f"  {nm:28s} n={len(y):5d} d={X.shape[1]}  {time.time() - t1:6.1f}s")

    if not all_runs:
        sys.exit("No datasets were run. Check --data-dir or use --quick.")

    df = pd.concat(all_runs, ignore_index=True)
    stab = pd.concat(all_stab, ignore_index=True)
    summary = summarise(df).merge(stab, on=["dataset", "m", "alpha", "adjacency", "method"])
    df.to_csv(out / "runs.csv", index=False)
    summary.to_csv(out / "summary_all_settings.csv", index=False)

    # ---- headline table: best (m, alpha) per dataset & method, per adjacency
    headline = []
    for adj in args.adjacency:
        best = best_settings(summary, adj)
        headline.append(best)
        fig_boxplot(df, best, adj, out / "figures" / f"boxplot_ari_{adj}.png")
        for nm in best.dataset.unique():
            Xn, y, labels = store[nm]
            if Xn.shape[1] == 2:
                safe = nm.replace("/", "_")
                fig_example(nm, Xn, y, df, labels, best, adj,
                            out / "figures" / f"example_{safe}_{adj}.png")
                b = best[(best.dataset == nm) & (best.method == "single")].iloc[0]
                fig_sweep(nm, Xn, y, int(b.m), b.alpha, args.k, adj,
                          out / "figures" / f"sweep_{safe}_{adj}.png")
    headline = pd.concat(headline, ignore_index=True)
    cols = ["dataset", "n", "d", "adjacency", "method", "m", "alpha", "ari_mean", "ari_std",
            "ari_min", "ari_max", "ari_range", "acc_mean", "acc_boundary_mean", "stability_ari",
            "noise_frac_mean", "n_clusters_min", "n_clusters_max"]
    headline = headline[cols]
    headline.to_csv(out / "summary_best.csv", index=False)

    # ---- console report + decision rules
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    show = ["dataset", "adjacency", "method", "m", "alpha", "ari_mean", "ari_std",
            "ari_range", "acc_boundary_mean", "stability_ari"]
    print("\n=== Best setting per dataset and method ===")
    print(headline[show].round(3).to_string(index=False))

    print("\n=== Decision check A: offset sensitivity and the resolution control ===")
    for adj in args.adjacency:
        h = headline[(headline.adjacency == adj)]
        t = {mt: h[h.method == mt].set_index("dataset").sort_index() for mt in METHODS}
        n_big = int((t["single"].ari_range > 0.10).sum())
        print(f"[{adj}] datasets with single-grid ARI range > 0.10 : {n_big} / {len(t['single'])}")
        for col, label in (("ari_mean", "mean ARI     "), ("ari_std", "ARI std      "),
                           ("ari_range", "ARI range    "),
                           ("acc_boundary_mean", "boundary acc.")):
            vals = "   ".join(f"{mt} = {t[mt][col].median():.3f}" for mt in METHODS)
            print(f"[{adj}] median {label}: {vals}")
        a, f = t["averaged"], t["single_fine"]
        win_mean = int((a.ari_mean > f.ari_mean + 0.01).sum())
        lose_mean = int((a.ari_mean < f.ari_mean - 0.01).sum())
        win_std = int((a.ari_std < f.ari_std).sum())
        print(f"[{adj}] averaged vs control: higher mean ARI on {win_mean}, lower on {lose_mean}, "
              f"tie (+/-0.01) on {len(a) - win_mean - lose_mean} datasets; "
              f"smaller ARI std on {win_std} / {len(a)}")
        # fairest single-grid baseline: best of coarse (m) and fine (m*k) per dataset
        best_any = np.maximum(t["single"].ari_mean, f.ari_mean)
        n_w = int((a.ari_mean > best_any + 0.01).sum())
        n_l = int((a.ari_mean < best_any - 0.01).sum())
        print(f"[{adj}] averaged vs BEST single grid at any tested resolution: higher on {n_w}, "
              f"lower on {n_l}, tie on {len(a) - n_w - n_l}; datasets where it loses: "
              f"{', '.join(a.index[a.ari_mean < best_any - 0.01]) or '-'}")
        frag = f[f.n_clusters_max > 5 * t["single"].n_clusters_max.clip(lower=1)].index
        if len(frag):
            print(f"[{adj}] note: control breaks clusters into fragments (>5x more clusters) on "
                  f"{len(frag)} datasets -> fine single grids are too sparse there")
        verdict = ("offset sensitivity looks REAL" if n_big >= max(1, len(t["single"]) // 3)
                   else "offset sensitivity looks SMALL -> rethink the topic")
        gain = ("averaging adds value BEYOND finer resolution"
                if win_mean > lose_mean and win_std > len(a) / 2 else
                "gain may be mostly due to finer resolution -> check before claiming it")
        print(f"[{adj}] verdict: {verdict}; {gain}\n")

    if all_base:
        base = pd.concat(all_base, ignore_index=True)
        base.to_csv(out / "baselines_all_settings.csv", index=False)
        bbest = base.loc[base.groupby(["dataset", "method"])["ari"].idxmax()]
        bbest.to_csv(out / "baselines_best.csv", index=False)
        print("=== Decision check A2: grid methods vs DBSCAN / HDBSCAN (best settings, mean ARI) ===")
        comp = []
        for adj in args.adjacency:
            h = headline[headline.adjacency == adj]
            tab = h.pivot(index="dataset", columns="method", values="ari_mean")
            for bm in ("dbscan", "hdbscan"):
                tab[bm] = bbest[bbest.method == bm].set_index("dataset").ari
            tab.insert(0, "adjacency", adj)
            comp.append(tab.reset_index())
            print(f"\n[{adj}]")
            print(tab.drop(columns="adjacency").round(3).to_string())
            for bm in ("dbscan", "hdbscan"):
                diff = tab["averaged"] - tab[bm]
                print(f"[{adj}] averaged vs {bm}: higher on {int((diff > 0.01).sum())}, "
                      f"lower on {int((diff < -0.01).sum())}, tie on {int((diff.abs() <= 0.01).sum())}; "
                      f"median difference = {diff.median():+.3f}")
        pd.concat(comp, ignore_index=True).to_csv(out / "compare_methods.csv", index=False)
        print("Note: DBSCAN/HDBSCAN use no grid, so they have no offset sensitivity at all.\n")

    if not args.no_boundary:
        print("Experiment B: known boundaries (rotated square, disk, annulus + noise)")
        bs = boundary_experiment(args, rng, out)
        print(bs.round(3).to_string(index=False))
        best_b = bs.loc[bs.groupby("method")["band_error_mean"].idxmin()].set_index("method")
        best_i = bs.loc[bs.groupby("method")["iou_mean"].idxmax()].set_index("method")
        print("\nBest setting per method (each method tuned separately):")
        for mt in METHODS:
            rb, ri = best_b.loc[mt], best_i.loc[mt]
            print(f"  {mt:12s} lowest band error = {rb.band_error_mean:.3f} (m={int(rb.m)}, "
                  f"alpha={rb.alpha})   highest IoU = {ri.iou_mean:.3f} +/- {ri.iou_std:.3f} "
                  f"(m={int(ri.m)}, alpha={ri.alpha})")
        better = best_b.loc["averaged", "band_error_mean"] < best_b.loc["single_fine", "band_error_mean"]
        print("verdict B: at their best settings, the averaged boundary is "
              + ("MORE accurate than the fine single grid" if better
                 else "NOT more accurate than the fine single grid") + "\n")
    if not args.no_scaling:
        print("Experiment C: runtime as N grows (each run in its own process, "
              f"timeout {args.time_budget}s)")
        scaling_experiment(args, out)
        print()
    print(f"Total time {time.time() - t0:.1f}s. Outputs in: {out.resolve()}")


if __name__ == "__main__":
    main()
