"""Post hoc comparison with a Misty-Mountain-like rule (Sugar & Sealfon, 2010) on the
locked list: one histogram (k=1) with the number of bins per axis chosen by Knuth's
Bayesian rule, and the Poisson peak-saddle criterion with z=2. Our union-find merge,
noise floor and 0.5% minimum size are reused, so this approximates, not reproduces,
Misty Mountain. Same datasets, loader and offsets as the locked run."""
import sys, os, numpy as np, pandas as pd
from scipy.special import gammaln
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import adjusted_rand_score as ari
from pilot_locked_test import LOCKED_TEST, load, SETTINGS
from pilot_grid_offset import normalise
from auto_core import floptics_auto, floptics_sa_sig

def knuth_m(Xn, mmax=100):
    N, d = Xn.shape; best, bm = -np.inf, 2
    for m in range(2, mmax + 1):
        M = m ** d
        if M > 4 * N: break
        idx = np.minimum((Xn * m).astype(np.int64), m - 1)
        n = np.bincount(np.ravel_multi_index(idx.T, (m,) * d), minlength=M)
        lp = (N * np.log(M) + gammaln(M / 2) - M * gammaln(0.5) - gammaln(N + M / 2)
              + gammaln(n + 0.5).sum())
        if lp > best: best, bm = lp, m
    return bm

if __name__ == "__main__":
    rng = np.random.RandomState(SETTINGS["seed"]); rows = []
    for nm in LOCKED_TEST:
        X, y = load(nm, sys.argv[1], SETTINGS["seed"]); Xn = normalise(X); d = Xn.shape[1]
        offs = rng.rand(SETTINGS["runs_per_grid_method"], d)
        mk = knuth_m(Xn)
        for name, f in (("misty_like", lambda s: floptics_sa_sig(Xn, mk, 1, s, 2.0, 1.0)),
                        ("misty_like_z1", lambda s: floptics_sa_sig(Xn, mk, 1, s, 1.0, 1.0)),
                        ("sa_knuth_z2", lambda s: floptics_sa_sig(Xn, mk, 4, s, 2.0, 1.0))):
            a = np.array([ari(y, f(s)) for s in offs])
            rows.append(dict(dataset=nm, method=name, m=mk, ari=a.mean(), sd=a.std()))
        print(nm, mk, flush=True)
    R = pd.DataFrame(rows); R.to_csv("misty_like_locked.csv", index=False)
    L = pd.read_csv("results_locked/locked_summary.csv") if os.path.exists("results_locked/locked_summary.csv") \
        else pd.read_csv("../results/locked_summary.csv")
    sa = L[L.method == "floptics_auto"].set_index("dataset").ari_mean
    print(R.groupby("method").agg(mean=("ari", "mean"), median=("ari", "median"), sd_median=("sd", "median")).round(3))
    for mth in R.method.unique():
        o = R[R.method == mth].set_index("dataset").ari.reindex(sa.index)
        diff = sa - o
        print(mth, "SA better/worse:", int((diff > .01).sum()), int((diff < -.01).sum()),
              "Wilcoxon p=%.4f" % stats.wilcoxon(sa, o).pvalue)
