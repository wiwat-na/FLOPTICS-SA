"""Theorem 2 audit on DLBCL FCS data (discretised channels): boundary fraction and
reflection equivariance with fixed m at s=0, s=1/(2k) and 5 random offsets."""
import sys, os, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import adjusted_rand_score as ari
from pilot_dlbcl import read_fcs
from pilot_grid_offset import normalise
from auto_core import floptics_auto, floptics_sa_sig
k = 4; rng = np.random.RandomState(7); rows = []
for f in sorted(glob.glob(sys.argv[1] + "/*.fcs")):
    X = read_fcs(f); Xn = normalise(X); d = 3
    nuniq = [np.unique(X[:, a]).size for a in range(d)]
    _, m = floptics_auto(Xn, np.zeros(d))
    offs = [("s=0", np.zeros(d)), ("s=1/(2k)", np.full(d, .5 / k))] + \
           [(f"rand{i}", rng.rand(d) / k) for i in range(5)]
    for nm, s in offs:
        u = Xn * m * k + k * s
        bf = (np.abs(u - np.round(u)) < 1e-6).any(axis=1).mean()
        base = floptics_sa_sig(Xn, m, k, s, 1.0, 1.0)
        sg = np.r_[(-s[0]) % (1 / k), s[1:]]
        r_exact = ari(base, floptics_sa_sig(np.c_[1 - Xn[:, :1], Xn[:, 1:]], m, k, sg, 1.0, 1.0))
        rows.append(dict(sample=os.path.basename(f)[:3], offset=nm, m=m, unique_values=min(nuniq),
                         boundary_frac=bf, ari_reflect_exact=r_exact))
R = pd.DataFrame(rows); R.to_csv("audit_thm2_dlbcl.csv", index=False)
R["kind"] = R.offset.where(~R.offset.str.startswith("rand"), "random")
print("min distinct values per channel:", R.unique_values.min(), "-", R.unique_values.max())
print(R.groupby("kind").agg(n=("ari_reflect_exact", "size"),
      exact=("ari_reflect_exact", lambda a: int((a > 1 - 1e-12).sum())),
      min_ari=("ari_reflect_exact", "min"), median_boundary=("boundary_frac", "median"),
      max_boundary=("boundary_frac", "max")).round(4).to_string())
