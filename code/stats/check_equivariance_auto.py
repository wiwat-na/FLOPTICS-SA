"""Does the automatic choice of m preserve B_d-equivariance? For each locked dataset,
compare the partition of X with that of an axis-permuted / reflected copy, at offset 0
(which every element of B_d maps to 0), with automatic m and with m fixed."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import adjusted_rand_score as ari
from pilot_locked_test import LOCKED_TEST, load
from pilot_grid_offset import normalise
from auto_core import floptics_auto, floptics_sa_sig
D = sys.argv[1]
rows = []
for nm in LOCKED_TEST:
    X, y = load(nm, D, 12345)
    Xn = normalise(X); z = np.zeros(X.shape[1])
    lab, m = floptics_auto(Xn, z)
    labf = floptics_sa_sig(Xn, m, 4, z, 1.0, 1.0)
    for g, Xg in (("swap", X[:, ::-1]), ("reflect0", X * np.r_[-1, np.ones(X.shape[1]-1)]),
                  ("reflect_all", -X)):
        Gn = normalise(Xg)
        lg, mg = floptics_auto(Gn, z)
        lgf = floptics_sa_sig(Gn, m, 4, z, 1.0, 1.0)
        rows.append(dict(dataset=nm, g=g, m=m, m_g=mg, ari_auto=ari(lab, lg), ari_fixed=ari(labf, lgf)))
    print(nm, rows[-1], flush=True)
R = pd.DataFrame(rows); R.to_csv("equivariance_auto.csv", index=False)
print("cases", len(R), "| m changed:", (R.m != R.m_g).sum(),
      "| fixed m ARI==1:", (R.ari_fixed > 1 - 1e-12).sum(),
      "| auto ARI==1:", (R.ari_auto > 1 - 1e-12).sum(), "| auto ARI min:", R.ari_auto.min().round(3))
print(R[R.m != R.m_g].to_string())
