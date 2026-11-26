#!/usr/bin/env python3
"""
FlowCAP-I DLBCL (30 FCS files) - real flow cytometry test.

Channels: FS, SS, FL1.LOG, FL2.LOG, FL4.LOG (already log-scaled by the instrument,
range 0-1023). The three fluorescence channels are used, as in the cytometree package
(CD3, CD5, CD19 according to FlowCAP-I; the FCS files do not say which channel is which).

Part 1 (no reference labels needed, all 30 samples)
  For every method and sample, run on R random grid offsets and measure
    stability   = mean ARI between the clusterings of every pair of offsets (1 = identical)
    worst_pair  = lowest ARI between two offsets
    n_clusters, seconds
Part 2 (reference labels)
  Every sample for which a label file is found: ARI and accuracy against manual gating.
  Label files: --labels-dir with one file per sample named like the FCS file
  (001.csv or 001.txt, one integer label per line, same order as the events).
  Sample 003 is also available with labels in the cytometree package (--cytometree).

Usage
  python pilot_dlbcl.py --fcs-dir DLBCL --cytometree Cytometree/data/DLBCL.RData
  python pilot_dlbcl.py --fcs-dir DLBCL --labels-dir DLBCL_labels
"""

import argparse
import glob
import itertools
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score

from pilot_grid_offset import normalise
from pilot_floptics import floptics, scores
from auto_core import floptics_auto

warnings.filterwarnings("ignore")
CHANNELS = ["FL1.LOG", "FL2.LOG", "FL4.LOG"]


def read_fcs(path):
    import flowio
    fd = flowio.FlowData(str(path))
    ev = np.reshape(np.array(fd.events, dtype=float), (-1, fd.channel_count))
    idx = [fd.pnn_labels.index(c) for c in CHANNELS]
    return ev[:, idx]


def read_labels(labels_dir, stem, cytometree):
    if labels_dir:
        for ext in (".csv", ".txt"):
            p = Path(labels_dir) / f"{stem}{ext}"
            if p.exists():
                v = pd.read_csv(p, header=None).iloc[:, -1].to_numpy()
                if not np.issubdtype(v.dtype, np.number):  # header line present
                    v = pd.read_csv(p).iloc[:, -1].to_numpy()
                return v.astype(int)
    if cytometree and stem == "003":
        import pyreadr
        return pyreadr.read_r(cytometree)["DLBCL"]["label"].to_numpy().astype(int)
    return None


METHODS = {
    "floptics_auto": lambda X, s: floptics_auto(X, s, k=4)[0],
    "floptics_auto_v1": lambda X, s: floptics_auto(X, s, k=4, m_rule="stable")[0],
    "floptics_auto_k1": lambda X, s: floptics_auto(X, s, k=1)[0],
    "floptics_b20": lambda X, s: floptics(X, 20, 10, s),
    "floptics_b50": lambda X, s: floptics(X, 50, 10, s),
    "hdbscan_1pct": lambda X, s: HDBSCAN(min_cluster_size=max(5, int(0.01 * len(X)))).fit_predict(X),
}
GRID_METHODS = [m for m in METHODS if not m.startswith("hdbscan")]  # updated in main()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fcs-dir", required=True)
    p.add_argument("--labels-dir", default=None)
    p.add_argument("--cytometree", default=None, help="path to cytometree DLBCL.RData")
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--methods", default=",".join(METHODS),
                   help="comma-separated subset of: " + ", ".join(METHODS))
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default="results_dlbcl")
    args = p.parse_args()
    chosen = [m for m in args.methods.split(",") if m]
    for m in list(METHODS):
        if m not in chosen:
            del METHODS[m]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    floptics_auto(np.random.rand(60, 3), np.zeros(3))  # compile once

    rows, lab_rows = [], []
    files = sorted(glob.glob(str(Path(args.fcs_dir) / "*.fcs")))
    t0 = time.time()
    for f in files:
        stem = Path(f).stem
        X = normalise(read_fcs(f))
        y = read_labels(args.labels_dir, stem, args.cytometree)
        offs = rng.rand(args.runs, X.shape[1])
        t1 = time.time()
        for method, fn in METHODS.items():
            n_runs = args.runs if not method.startswith("hdbscan") else 1
            labs, secs = [], []
            for r in range(n_runs):
                t = time.perf_counter()
                labs.append(fn(X, offs[r]))
                secs.append(time.perf_counter() - t)
            pairs = [adjusted_rand_score(a, b) for a, b in itertools.combinations(labs, 2)]
            ncl = [len(set(lb.tolist()) - {-1}) for lb in labs]
            rows.append(dict(sample=stem, n=len(X), method=method,
                             stability=float(np.mean(pairs)) if pairs else np.nan,
                             worst_pair=float(np.min(pairs)) if pairs else np.nan,
                             n_clusters_min=min(ncl), n_clusters_max=max(ncl),
                             seconds=float(np.median(secs))))
            if y is not None:
                sc = [scores(y, lb) for lb in labs]
                lab_rows.append(dict(sample=stem, method=method,
                                     ari_mean=np.mean([s["ari"] for s in sc]),
                                     ari_std=np.std([s["ari"] for s in sc]),
                                     ari_min=np.min([s["ari"] for s in sc]),
                                     acc_mean=np.mean([s["paper_accuracy"] for s in sc])))
        print(f"  {stem} n={len(X):6d} labels={'yes' if y is not None else 'no '} "
              f"{time.time() - t1:6.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(out / "dlbcl_stability.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n=== Part 1: stability across grid offsets (median over samples) ===")
    agg = df.groupby("method").agg(stability_median=("stability", "median"),
                                   stability_min=("stability", "min"),
                                   worst_pair_median=("worst_pair", "median"),
                                   samples_below_0_9=("stability", lambda v: int((v < 0.9).sum())),
                                   clusters_median=("n_clusters_max", "median"),
                                   seconds_median=("seconds", "median"))
    print(agg.loc[list(METHODS)].round(3).to_string())
    print("\nper-sample stability:")
    print(df.pivot(index="sample", columns="method", values="stability")
          [[m for m in METHODS if not m.startswith("hdbscan")]]
          .round(3).to_string())
    if lab_rows:
        dl = pd.DataFrame(lab_rows)
        dl.to_csv(out / "dlbcl_vs_manual_gating.csv", index=False)
        print(f"\n=== Part 2: agreement with manual gating ({dl['sample'].nunique()} samples) ===")
        print(dl.groupby("method")[["ari_mean", "ari_std", "ari_min", "acc_mean"]].median()
              .loc[[m for m in METHODS if m in set(dl.method)]].round(3).to_string())
        if dl["sample"].nunique() <= 5:
            print(dl.round(3).to_string(index=False))
    print(f"\nTotal time {time.time() - t0:.1f}s. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
