"""Theorem 2 audit on the locked datasets at 5 random offsets per dataset (fixed m),
reflection applied exactly (x -> 1-x on normalised data) and via raw data + min-max."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import adjusted_rand_score as ari
from pilot_locked_test import LOCKED_TEST, load
from pilot_grid_offset import normalise
from auto_core import floptics_auto, floptics_sa_sig
k = 4; rng = np.random.RandomState(11); rows = []
for nm in LOCKED_TEST:
    X, _ = load(nm, sys.argv[1], 12345); d = X.shape[1]; Xn = normalise(X)
    _, m = floptics_auto(Xn, np.zeros(d))
    for i in range(5):
        s = rng.rand(d) / k
        u = Xn * m * k + k * s
        bf = (np.abs(u - np.round(u)) < 1e-6).any(axis=1).mean()
        base = floptics_sa_sig(Xn, m, k, s, 1.0, 1.0)
        sg = np.r_[(-s[0]) % (1 / k), s[1:]]
        e = ari(base, floptics_sa_sig(np.c_[1 - Xn[:, :1], Xn[:, 1:]], m, k, sg, 1.0, 1.0))
        r = ari(base, floptics_sa_sig(normalise(X * np.r_[-1, np.ones(d - 1)]), m, k, sg, 1.0, 1.0))
        p = ari(base, floptics_sa_sig(normalise(X[:, ::-1]), m, k, s[::-1], 1.0, 1.0))
        rows.append(dict(dataset=nm, run=i, boundary_frac=bf, exact=e, raw=r, swap=p))
R = pd.DataFrame(rows); R.to_csv("audit_thm2_random.csv", index=False)
for c in ("exact", "raw", "swap"):
    print(c, int((R[c] > 1 - 1e-12).sum()), "/", len(R), "min", round(R[c].min(), 4))
print("boundary frac max", R.boundary_frac.max())
