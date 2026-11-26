"""Automatic m on hard synthetic cases with known labels: unequal densities, unequal sizes,
background noise, nested (two-level) clusters. Reports ARI of automatic m and of the
oracle m (best ladder value in hindsight), 5 replicates each, random offsets."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import adjusted_rand_score as ari
from pilot_grid_offset import normalise
from auto_core import floptics_sa_sig, stable_m_fixed, M_LADDER
k = 4; rng = np.random.RandomState(5)
def blobs(spec, noise_frac=0.0):
    X, y = [], []
    for lab, (c, sd, n) in enumerate(spec):
        X.append(rng.randn(n, len(c)) * sd + np.array(c)); y += [lab] * n
    X = np.vstack(X); y = np.array(y)
    if noise_frac:
        nn = int(noise_frac * len(X) / (1 - noise_frac)); lo, hi = X.min(0) - 1, X.max(0) + 1
        X = np.vstack([X, rng.rand(nn, X.shape[1]) * (hi - lo) + lo]); y = np.r_[y, -np.ones(nn, int)]
    return X, y
S = {
 "unequal_density": lambda: blobs([((0, 0), 1.0, 9000), ((5, 0), 0.15, 1000), ((0, 5), 0.15, 300)]),
 "unequal_size_100to1": lambda: blobs([((0, 0), .5, 10000), ((3, 0), .5, 1000), ((0, 3), .5, 100)]),
 "noise_0": lambda: blobs([((0, 0), .5, 3000), ((3, 0), .5, 3000), ((0, 3), .5, 3000)]),
 "noise_20": lambda: blobs([((0, 0), .5, 3000), ((3, 0), .5, 3000), ((0, 3), .5, 3000)], .2),
 "noise_50": lambda: blobs([((0, 0), .5, 3000), ((3, 0), .5, 3000), ((0, 3), .5, 3000)], .5),
 "nested_2_levels": lambda: blobs([((0, 0), .3, 2500), ((1.6, 0), .3, 2500), ((8, 0), .3, 2500), ((9.6, 0), .3, 2500)]),
 "3d_unequal_density": lambda: blobs([((0, 0, 0), 1.0, 9000), ((5, 0, 0), .2, 1000), ((0, 5, 0), .2, 300)]),
}
rows = []
for name, gen in S.items():
    for r in range(5):
        X, y = gen(); Xn = normalise(X); N, d = Xn.shape; s = rng.rand(d) / k
        lad = [m for m in M_LADDER if m ** d <= N and (m * k + 2 * k + 4) ** d <= 3e7]
        labs = {m: floptics_sa_sig(Xn, m, k, s, 1.0, 1.0) for m in lad}
        ev = lambda lab: ari(y, lab) if not (y < 0).any() else ari(y[y >= 0], lab[y >= 0])
        ar = {m: ev(l) for m, l in labs.items()}
        m_auto = stable_m_fixed(Xn, k, 1.0, 1.0); m_or = max(ar, key=ar.get)
        row = dict(scenario=name, rep=r, m_auto=m_auto, m_oracle=m_or, ari_auto=ar[m_auto],
                   ari_oracle=ar[m_or], k_auto=len(set(labs[m_auto]) - {-1}))
        if name == "nested_2_levels":
            yc = (y >= 2).astype(int)
            row["ari_auto_coarse_truth"] = ari(yc, labs[m_auto])
        rows.append(row)
    print(name, flush=True)
R = pd.DataFrame(rows); R.to_csv("audit_m_synthetic.csv", index=False)
print(R.groupby("scenario")[["ari_auto", "ari_oracle", "k_auto", "m_auto", "m_oracle"]].median().round(3).to_string())
if "ari_auto_coarse_truth" in R: print("nested: ARI vs 2-cluster truth", R.ari_auto_coarse_truth.dropna().round(3).tolist())
