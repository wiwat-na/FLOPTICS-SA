"""Friedman test, Iman-Davenport correction, Nemenyi critical difference and Wilcoxon
signed-rank tests (Holm) of FLOPTICS-SA against every other method on the locked list.
Uses results_locked/locked_summary.csv (mean ARI over offsets per dataset). Post hoc
analysis of the frozen results: no method was rerun."""
import numpy as np, pandas as pd
from scipy import stats
d = pd.read_csv("../results/locked_summary.csv" if __import__("os").path.exists("../results/locked_summary.csv") else "results_locked/locked_summary.csv")
P = d.pivot(index="dataset", columns="method", values="ari_mean")
assert P.shape == (38, 11) and not P.isna().any().any()
N, k = P.shape
R = P.rank(axis=1, ascending=False, method="average")   # rank 1 = best
mr = R.mean().sort_values()
chi2, pF = stats.friedmanchisquare(*[P[c].values for c in P.columns])
FF = (N - 1) * chi2 / (N * (k - 1) - chi2)
pFF = stats.f.sf(FF, k - 1, (k - 1) * (N - 1))
q05 = {11: 3.219}[k]                                     # Studentised range / sqrt(2), Demsar 2006
CD = q05 * np.sqrt(k * (k + 1) / (6 * N))
print(f"N={N} k={k} Friedman chi2={chi2:.2f} p={pF:.2e}; Iman-Davenport F={FF:.2f} p={pFF:.2e}; Nemenyi CD(0.05)={CD:.2f}")
print(mr.round(2).to_string())
ref = "floptics_auto"
rows = []
for c in P.columns:
    if c == ref: continue
    diff = P[ref] - P[c]
    w = stats.wilcoxon(P[ref], P[c], zero_method="wilcox")
    rows.append(dict(method=c, mean_diff=diff.mean(), median_diff=diff.median(),
                     wins=(diff > 0.01).sum(), losses=(diff < -0.01).sum(), p=w.pvalue,
                     rank_gap=mr[c] - mr[ref]))
W = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
m = len(W); adj = []; running = 0
for i, p in enumerate(W.p):
    running = max(running, min(1, (m - i) * p)); adj.append(running)
W["p_holm"] = adj
W["nemenyi_sig"] = W.rank_gap > CD
print(W.round(4).to_string())
W.to_csv("locked_wilcoxon_holm.csv", index=False)
mr.to_csv("locked_mean_ranks.csv")
