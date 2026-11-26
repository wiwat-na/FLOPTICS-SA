"""Automatic m against the best m in hindsight (oracle) on the locked list, and against
Scott's and Knuth's rules. ARI of every ladder value is averaged over 3 random offsets."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import adjusted_rand_score as ari
from pilot_locked_test import LOCKED_TEST, load, SETTINGS
from pilot_grid_offset import normalise
from auto_core import floptics_sa_sig, stable_m_fixed, scott_m, M_LADDER
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from misty_like import knuth_m
k = 4; rng = np.random.RandomState(99); rows = []
for nm in LOCKED_TEST:
    X, y = load(nm, sys.argv[1], SETTINGS["seed"]); Xn = normalise(X); N, d = Xn.shape
    offs = rng.rand(3, d)
    lad = [m for m in M_LADDER if m ** d <= N]
    ar = {m: np.mean([ari(y, floptics_sa_sig(Xn, m, k, s, 1.0, 1.0)) for s in offs]) for m in lad}
    m_auto = stable_m_fixed(Xn, k, 1.0, 1.0); m_sc = scott_m(Xn); m_kn = knuth_m(Xn)
    snap = lambda m: min(lad, key=lambda v: abs(v - m))
    m_or = max(ar, key=ar.get)
    rows.append(dict(dataset=nm, N=N, m_auto=m_auto, m_oracle=m_or, m_scott=m_sc, m_knuth=m_kn,
                     ari_auto=ar[m_auto], ari_oracle=ar[m_or], ari_scott=ar[snap(m_sc)],
                     ari_knuth=ar[snap(m_kn)], rank_auto=sorted(ar.values(), reverse=True).index(ar[m_auto]) + 1,
                     n_ladder=len(lad)))
    print(nm, rows[-1]["m_auto"], m_or, round(rows[-1]["ari_auto"], 3), round(ar[m_or], 3), flush=True)
R = pd.DataFrame(rows); R.to_csv("audit_m_oracle.csv", index=False)
R["regret"] = R.ari_oracle - R.ari_auto
print(R[["ari_auto", "ari_oracle", "ari_scott", "ari_knuth"]].agg(["mean", "median"]).round(3).to_string())
print("regret median %.3f, <=0.05 in %d/38, >0.2 in %d/38; rank of auto m median %d of median ladder %d"
      % (R.regret.median(), (R.regret <= 0.05).sum(), (R.regret > 0.2).sum(), R.rank_auto.median(), R.n_ladder.median()))
print(R.sort_values("regret", ascending=False).head(6)[["dataset", "m_auto", "m_oracle", "ari_auto", "ari_oracle"]].to_string())
