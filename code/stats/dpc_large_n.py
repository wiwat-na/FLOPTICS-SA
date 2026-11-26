"""Post hoc sensitivity check for density peaks (DPC) on the locked list.
In the locked run DPC was skipped (all points labelled noise, ARI 0) on the three datasets
with N > 8,000 because baselines_free.dpc has a memory guard max_n=8000. This script runs the
same DPC, without the guard, on those three datasets and recomputes the locked statistics
with (a) these ARIs and (b) the three datasets removed. The locked results are not changed."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot_auto_floptics import load
from pilot_floptics import scores
from pilot_grid_offset import normalise
from baselines_free import dpc

import os
def _p(a, b): return a if os.path.exists(a) else b
DATA = sys.argv[1] if len(sys.argv) > 1 else "clustering-data-v1"
BIG = ["other/chameleon_t7_10k", "wut/isolation", "wut/trajectories"]
new = {}
for nm in BIG:
    X, y = load(nm, DATA, 2026)
    t = time.perf_counter(); lab = dpc(normalise(X), max_n=10**9)
    new[nm] = scores(y, lab)["ari"]
    print(f"{nm:24s} n={len(y)} DPC ARI={new[nm]:.3f} ({time.perf_counter()-t:.1f}s)")

d = pd.read_csv(_p("../results/locked_summary.csv", "results_locked/locked_summary.csv"))
P0 = d.pivot(index="dataset", columns="method", values="ari_mean")
def report(P, tag):
    N, k = P.shape
    R = P.rank(axis=1, ascending=False, method="average"); mr = R.mean()
    chi2, _ = stats.friedmanchisquare(*[P[c].values for c in P.columns])
    CD = 3.219 * np.sqrt(k * (k + 1) / (6 * N))
    rows = []
    for c in P.columns:
        if c == "floptics_auto": continue
        rows.append(dict(method=c, p=stats.wilcoxon(P["floptics_auto"], P[c]).pvalue,
                         gap=mr[c] - mr["floptics_auto"]))
    W = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    run = 0; adj = []
    for i, p in enumerate(W.p):
        run = max(run, min(1, (len(W) - i) * p)); adj.append(run)
    W["p_holm"] = adj
    print(f"\n[{tag}] N={N} mean ARI SA={P['floptics_auto'].mean():.3f} DPC={P['dpc'].mean():.3f}"
          f" | rank SA={mr['floptics_auto']:.2f} DPC={mr['dpc']:.2f} best={mr.idxmin()}"
          f" | Friedman chi2={chi2:.2f} CD={CD:.2f}")
    print(W.round(4).to_string())
    return W.assign(setting=tag)
P1 = P0.copy()
for nm, a in new.items(): P1.loc[nm, "dpc"] = a
out = pd.concat([report(P0, "locked"), report(P1, "dpc_without_guard"),
                 report(P0.drop(BIG), "without_3_large")])
out.to_csv(_p("../results", ".") + "/dpc_large_n.csv", index=False)
pd.Series(new).to_csv(_p("../results", ".") + "/dpc_large_n_ari.csv")

# Development list: datasets A-D (N = 10,000) were skipped in the same way.
dev = pd.read_csv(_p("../results/auto_summary_with_kde_automato.csv", "auto_summary_with_kde_automato.csv"))
newdev = {}
for L in "ABCD":
    X, y = load("paper/" + L, DATA, 2026)
    newdev["paper/" + L] = scores(y, dpc(normalise(X), max_n=10**9))["ari"]
    print(f"paper/{L} DPC ARI={newdev['paper/' + L]:.3f}")
for split in ("dev", "test"):
    s = dev[(dev.split == split) & (dev.method == "dpc")].set_index("dataset").ari_mean.copy()
    if len(s) == 0: continue
    m0, md0 = s.mean(), s.median()
    for nm, a in newdev.items():
        if nm in s.index: s[nm] = a
    print(f"{split}: DPC mean {m0:.3f}->{s.mean():.3f}, median {md0:.3f}->{s.median():.3f} (n={len(s)})")
pd.Series(newdev).to_csv(_p("../results", ".") + "/dpc_large_n_dev_ari.csv")
