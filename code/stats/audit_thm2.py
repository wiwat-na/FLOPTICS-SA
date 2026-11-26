"""Audit of Theorem 2 on the 38 locked datasets with fixed m.
For each dataset: fraction of points on a fine-cell boundary, and ARI between the
partition of X and of transformed copies, at offset s=0 and s=1/(2k) on every axis.
Reflections are applied (a) to raw data, then min-max normalised as in the code, and
(b) directly to the normalised data as x -> 1-x (the setting of the theorem)."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import adjusted_rand_score as ari
from pilot_locked_test import LOCKED_TEST, load
from pilot_grid_offset import normalise
from auto_core import floptics_auto, floptics_sa_sig
D = sys.argv[1]; k = 4
rows = []
for nm in LOCKED_TEST:
    X, y = load(nm, D, 12345); d = X.shape[1]
    Xn = normalise(X)
    _, m = floptics_auto(Xn, np.zeros(d))
    for sname, s in (("s=0", np.zeros(d)), ("s=1/(2k)", np.full(d, 0.5 / k))):
        u = Xn * m * k + k * s
        on_b = (np.abs(u - np.round(u)) < 1e-6).any(axis=1).mean()
        base = floptics_sa_sig(Xn, m, k, s, 1.0, 1.0)
        refl = np.r_[-1, np.ones(d - 1)]
        cases = {"swap": (normalise(X[:, ::-1]), s[::-1]),
                 "reflect0_raw": (normalise(X * refl), np.r_[(-s[0]) % (1 / k), s[1:]]),
                 "reflect0_exact": (np.c_[1 - Xn[:, :1], Xn[:, 1:]], np.r_[(-s[0]) % (1 / k), s[1:]])}
        for g, (Xg, sg) in cases.items():
            rows.append(dict(dataset=nm, offset=sname, g=g, m=m, boundary_frac=on_b,
                             ari=ari(base, floptics_sa_sig(Xg, m, k, sg, 1.0, 1.0))))
R = pd.DataFrame(rows); R.to_csv("audit_thm2.csv", index=False)
S = R.groupby(["offset", "g"]).agg(exact=("ari", lambda a: int((a > 1 - 1e-12).sum())),
                                    n=("ari", "size"), min_ari=("ari", "min"))
print(S.to_string())
print(R.groupby("offset").boundary_frac.describe().round(4).to_string())
