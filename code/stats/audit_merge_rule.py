"""Behaviour of the significance merge with the full automatic method.
Null: one cluster (uniform square/cube, isotropic Gaussian, elongated Gaussian, and a
Gaussian rounded to 256 levels per axis as in instrument data). Power: two equal
Gaussians at separation Delta (in SD). Reports the number of clusters for z in {0.5,1,2,3}."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import adjusted_rand_score as ari
from pilot_grid_offset import normalise
from auto_core import floptics_auto
ZS = (0.5, 1.0, 2.0, 3.0); k = 4
def null_data(kind, n, d, rng):
    if kind == "uniform": return rng.rand(n, d)
    if kind == "gauss": return rng.randn(n, d)
    if kind == "elongated": return rng.randn(n, d) * np.r_[1.0, np.full(d - 1, 0.3)]
    if kind == "gauss_rounded":
        X = rng.randn(n, d); lo, hi = -4, 4
        return np.round((np.clip(X, lo, hi) - lo) / (hi - lo) * 255)
rows = []
rng = np.random.RandomState(2026)
for d in (2, 3):
    for kind in ("uniform", "gauss", "elongated", "gauss_rounded"):
        for n in (2000, 10000, 50000):
            for r in range(10):
                X = null_data(kind, n, d, rng); Xn = normalise(X); s = rng.rand(d) / k
                for z in ZS:
                    lab, m = floptics_auto(Xn, s, z=z)
                    rows.append(dict(exp="null", d=d, kind=kind, n=n, rep=r, z=z, m=m,
                                     n_clusters=len(set(lab) - {-1}), noise=(lab == -1).mean()))
        print("null done", d, kind, flush=True)
for d in (2,):
    for delta in (2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
        for n in (2000, 10000):
            for r in range(10):
                c = np.zeros(d); c[0] = delta
                X = np.r_[rng.randn(n // 2, d), rng.randn(n // 2, d) + c]
                y = np.r_[np.zeros(n // 2), np.ones(n // 2)]
                Xn = normalise(X); s = rng.rand(d) / k
                for z in ZS:
                    lab, m = floptics_auto(Xn, s, z=z)
                    rows.append(dict(exp="power", d=d, kind=f"delta={delta}", n=n, rep=r, z=z, m=m,
                                     n_clusters=len(set(lab) - {-1}), ari=ari(y, lab)))
        print("power done", delta, flush=True)
R = pd.DataFrame(rows); R.to_csv("audit_merge_rule.csv", index=False)
N = R[R.exp == "null"].assign(split=lambda t: t.n_clusters > 1)
print(N.groupby(["d", "kind", "z"]).split.mean().unstack("z").round(2).to_string())
P = R[R.exp == "power"].assign(found2=lambda t: t.n_clusters == 2)
print(P.groupby(["kind", "n", "z"]).found2.mean().unstack("z").round(2).to_string())
