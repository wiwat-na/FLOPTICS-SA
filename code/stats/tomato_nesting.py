"""Are the partitions nested? Coarser(A, B) = every cluster of B lies inside one cluster of A
(noise ignored). Checks ToMATo(z) coarser than Eq. (3), and Eq. (3) coarser than ToMATo(z/sqrt2)."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tomato_sandwich import run, LOCKED_TEST, load, SETTINGS, normalise, stable_m_fixed
def coarser(A, B):
    ok = (A >= 0) & (B >= 0)
    for b in np.unique(B[ok]):
        if np.unique(A[ok & (B == b)]).size > 1: return False
    return True
rng = np.random.RandomState(3); rows = []
for nm in LOCKED_TEST:
    X, y = load(nm, sys.argv[1], SETTINGS["seed"]); Xn = normalise(X); d = Xn.shape[1]
    m = stable_m_fixed(Xn, 4, 1.0, 1.0)
    for i in range(2):
        s = rng.rand(d) / 4
        h, e, l = (run(Xn, m, 4, s, md)[0] for md in ("tom_high", "eq3", "tom_low"))
        rows.append(dict(dataset=nm, high_coarser_eq3=coarser(h, e), eq3_coarser_low=coarser(e, l)))
R = pd.DataFrame(rows); R.to_csv("tomato_nesting.csv", index=False)
print("ToMATo(z) coarser than Eq.3:", int(R.high_coarser_eq3.sum()), "/", len(R),
      "| Eq.3 coarser than ToMATo(z/sqrt2):", int(R.eq3_coarser_low.sum()), "/", len(R))
print(R[~(R.high_coarser_eq3 & R.eq3_coarser_low)].to_string())
