"""Whole-run check of the relation between Eq. (3) and ToMATo on sqrt(D). The per-test
bounds do not by themselves order complete runs, because each merge changes later states.
We run the same procedure three times with only the merge test changed:
  eq3        p - l <= z sqrt(p + l)
  tom_low    sqrt(p) - sqrt(l) <= z / sqrt(2)
  tom_high   sqrt(p) - sqrt(l) <= z
and compare numbers of clusters (before the 0.5% size rule) and partitions."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import adjusted_rand_score as ari
from pilot_locked_test import LOCKED_TEST, load, SETTINGS
from pilot_grid_offset import normalise, grid_counts, triangle_smooth
from pilot_invariant_floptics import chebyshev_offsets, _find
from auto_core import stable_m_fixed

def merge(flat, order, offsets, key2, mode, z):
    n = flat.size; parent = np.full(n, -1, np.int64); peak = np.zeros(n); peak2 = np.zeros(n)
    for c in order:
        roots = []
        for o in offsets:
            q = c + o
            if 0 <= q < n and parent[q] != -1:
                r = _find(parent, q)
                if r not in roots: roots.append(r)
        if not roots:
            parent[c] = c; peak[c] = flat[c]; peak2[c] = key2[c]; continue
        best = roots[0]
        for r in roots[1:]:
            if peak[r] > peak[best] or (peak[r] == peak[best] and peak2[r] > peak2[best]): best = r
        parent[c] = best; l = flat[c]
        for r in roots:
            if r == best: continue
            p = peak[r]
            if mode == "eq3": m = p - l <= z * np.sqrt(p + l)
            elif mode == "tom_low": m = np.sqrt(p) - np.sqrt(l) <= z / np.sqrt(2)
            else: m = np.sqrt(p) - np.sqrt(l) <= z
            if m: parent[r] = best
    return parent

def run(Xn, m, k, s, mode, z=1.0):
    N, d = Xn.shape
    c, f = grid_counts(Xn, m * k, s * k, pad=k + 1)
    D = triangle_smooth(c, k).copy()
    for ax in range(d):
        sl = [slice(None)] * d; sl[ax] = 0; D[tuple(sl)] = 0; sl[ax] = -1; D[tuple(sl)] = 0
    flat = D.ravel()
    v = ((Xn - 0.5 * (1 - 1e-9)) ** 2).sum(axis=1)
    key2 = triangle_smooth(np.bincount(f, weights=v, minlength=flat.size).reshape(D.shape), k).ravel()
    cand = np.flatnonzero(flat >= 1.0)
    order = cand[np.lexsort((-key2[cand], -flat[cand]))]
    par = merge(flat, order, chebyshev_offsets(D.shape), key2, mode, z)
    roots = np.array([_find(par, i) if par[i] != -1 else -1 for i in range(flat.size)])
    lab = roots[f]
    return lab, len(set(lab.tolist()) - {-1})

if __name__ == "__main__":
    rng = np.random.RandomState(3); rows = []
    for nm in LOCKED_TEST:
        X, y = load(nm, sys.argv[1], SETTINGS["seed"]); Xn = normalise(X); d = Xn.shape[1]
        m = stable_m_fixed(Xn, 4, 1.0, 1.0)
        for i in range(2):
            s = rng.rand(d) / 4
            L = {md: run(Xn, m, 4, s, md) for md in ("tom_high", "eq3", "tom_low")}
            rows.append(dict(dataset=nm, run=i, K_high=L["tom_high"][1], K_eq3=L["eq3"][1], K_low=L["tom_low"][1],
                             ari_eq3_high=ari(L["eq3"][0], L["tom_high"][0]), ari_eq3_low=ari(L["eq3"][0], L["tom_low"][0])))
        print(nm, rows[-1]["K_high"], rows[-1]["K_eq3"], rows[-1]["K_low"], flush=True)
    R = pd.DataFrame(rows); R.to_csv("tomato_sandwich.csv", index=False)
    ok = ((R.K_high <= R.K_eq3) & (R.K_eq3 <= R.K_low))
    print("K_high <= K_eq3 <= K_low in", int(ok.sum()), "of", len(R), "runs")
    print(R[~ok].to_string())
