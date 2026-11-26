"""(a) Background noise: where do the uniform noise points go, and how big are the extra
clusters? (b) Large N: m chosen on a 20,000-point subsample and rescaled by
(N/20000)^(1/(d+4)) versus m chosen on all points, with the oracle m for reference."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import adjusted_rand_score as ari
from pilot_grid_offset import normalise, make_scaling_data
from auto_core import floptics_sa_sig, stable_m_fixed, floptics_auto, M_LADDER
k = 4; rng = np.random.RandomState(8)
print("(a) noise")
for frac in (0.0, 0.2, 0.5):
    for r in range(5):
        c = [(0, 0), (3, 0), (0, 3)]
        X = np.vstack([rng.randn(3000, 2) * .5 + np.array(cc) for cc in c]); y = np.repeat([0, 1, 2], 3000)
        nn = int(frac * 9000 / (1 - frac)) if frac else 0
        if nn:
            lo, hi = X.min(0) - 1, X.max(0) + 1
            X = np.vstack([X, rng.rand(nn, 2) * (hi - lo) + lo]); y = np.r_[y, -np.ones(nn, int)]
        Xn = normalise(X); lab, m = floptics_auto(Xn, rng.rand(2) / k)
        cl = [c_ for c_ in set(lab) - {-1}]
        # a cluster is "true" if most of its points are signal points
        true = [c_ for c_ in cl if (y[lab == c_] >= 0).mean() > 0.5]
        extra = [c_ for c_ in cl if c_ not in true]
        print(f"noise={frac:.0%} rep={r} m={m} clusters={len(cl)} true={len(true)} extra={len(extra)}"
              f" noise_pts_in_clusters={((lab >= 0) & (y < 0)).sum() / max(nn, 1):.2f}"
              f" extra_sizes_pct={[round(100 * (lab == c_).mean(), 1) for c_ in extra]}"
              f" ARI_all(noise as label)={ari(y, lab):.3f}", flush=True)
print("(b) rescale")
rows = []
for d, Ns in ((2, (100000, 300000)), (3, (100000,))):
    for N in Ns:
        X, y = make_scaling_data(N, d, 2026 + N); Xn = normalise(X); s = rng.rand(d) / k
        _, m_resc = floptics_auto(Xn, s)
        m_full = stable_m_fixed(Xn, k, 1.0, 1.0)
        idx = np.random.RandomState(12345).choice(N, 20000, replace=False)
        m_sub = stable_m_fixed(Xn[idx], k, 1.0, 1.0)
        lad = [m for m in M_LADDER if (m * k + 2 * k + 4) ** d <= 3e7]
        ar = {m: ari(y, floptics_sa_sig(Xn, m, k, s, 1.0, 1.0)) for m in sorted(set(lad) | {m_resc})}
        mo = max(ar, key=ar.get)
        rows.append(dict(d=d, N=N, m_sub=m_sub, m_rescaled=m_resc, m_full=m_full, m_oracle=mo,
                         ari_rescaled=ar[m_resc], ari_full=ar.get(m_full), ari_sub_unscaled=ar.get(m_sub),
                         ari_oracle=ar[mo]))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("audit_m_rescale.csv", index=False)
