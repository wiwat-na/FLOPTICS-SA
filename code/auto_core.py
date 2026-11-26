"""
FLOPTICS-SA with automatic parameters (core functions).

Removes the two tuned parameters of FLOPTICS-SA:

  beta (persistence)  ->  a significance test on counts.
      The averaged density D is measured in "points per coarse bin". If counts behave
      like Poisson counts, a peak of height p seen from a connecting level l differs from
      noise only if  p - l  >  z * sqrt(p + l).  A younger group is merged into its elder
      neighbour unless its peak passes this test. z = 3 by default ("3 standard deviations").
      The rule adapts to N on its own: more data -> larger counts -> smaller relative
      fluctuations -> finer structure becomes significant.

  m (bins per axis)   ->  chosen by stability across resolutions (default) or by a
      normal-reference rule.
      'stable': run the method for a ladder of m values; pick the m whose clustering agrees
               best (mean ARI) with the clusterings at the neighbouring m values.
               No reference labels are used.
      'scott' : h = 3.5 * mean(std per axis) * N^(-1/(d+2))  (Scott's histogram rule),
               m = round(1/h).

  floor: cells whose averaged count is below `floor_count` points per coarse bin are noise
         (default 1.0, i.e. "empty on average", the analogue of FLOPTICS' non-empty bins).
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score

from pilot_grid_offset import grid_counts, triangle_smooth
from pilot_invariant_floptics import chebyshev_offsets, _find, njit

M_LADDER = (4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50)


@njit(cache=True)
def _significance_merge(flat, order, offsets, z, key2):
    n = flat.size
    parent = np.full(n, -1, dtype=np.int64)
    peak = np.zeros(n)
    peak2 = np.zeros(n)
    roots = np.empty(offsets.size, dtype=np.int64)
    for c in order:
        nr = 0
        for o in offsets:
            q = c + o
            if parent[q] != -1:
                r = _find(parent, q)
                dup = False
                for j in range(nr):
                    if roots[j] == r:
                        dup = True
                        break
                if not dup:
                    roots[nr] = r
                    nr += 1
        if nr == 0:
            parent[c] = c
            peak[c] = flat[c]
            peak2[c] = key2[c]
            continue
        best = roots[0]
        for j in range(1, nr):
            rj = roots[j]
            if peak[rj] > peak[best] or (peak[rj] == peak[best] and peak2[rj] > peak2[best]):
                best = rj
        parent[c] = best
        lvl = flat[c]
        for j in range(nr):
            r = roots[j]
            if r != best:
                p = peak[r]
                if p - lvl <= z * np.sqrt(p + lvl):  # not significant -> merge
                    parent[r] = best
    for i in range(n):
        if parent[i] != -1:
            parent[i] = _find(parent, i)
    return parent


def floptics_sa_sig(Xn, m, k, offset, z=3.0, floor_count=1.0, min_frac=0.005):
    N, d = Xn.shape
    c, f = grid_counts(Xn, m * k, offset * k, pad=k + 1)
    D = (triangle_smooth(c, k) if k > 1 else c).copy()
    for ax in range(d):
        sl = [slice(None)] * d
        sl[ax] = 0
        D[tuple(sl)] = 0
        sl[ax] = -1
        D[tuple(sl)] = 0
    flat = D.ravel()
    # Secondary key that breaks density ties symmetrically: the same triangle filter applied
    # to sum ||x - centre||^2 per fine cell (centre of the normalised box). It is continuous
    # and unchanged by axis permutations/reflections, so the visiting order no longer depends
    # on cell indices (needed for exact B_d equivariance, Theorem 1b).
    v = ((Xn - 0.5 * (1 - 1e-9)) ** 2).sum(axis=1)
    sv = np.bincount(f, weights=v, minlength=flat.size).reshape(D.shape)
    key2 = (triangle_smooth(sv, k) if k > 1 else sv).ravel()
    cand = np.flatnonzero(flat >= floor_count)
    order = cand[np.lexsort((-key2[cand], -flat[cand]))]
    parent = _significance_merge(flat, order, chebyshev_offsets(D.shape), z, key2)
    rp = parent[f]
    lab = np.full(N, -1)
    ok = rp != -1
    if ok.any():
        _, lab[ok] = np.unique(rp[ok], return_inverse=True)
        sizes = np.bincount(lab[ok])
        small = np.flatnonzero(sizes < max(2, min_frac * N))
        lab[np.isin(lab, small)] = -1
        keep = lab != -1
        if keep.any():
            _, lab[keep] = np.unique(lab[keep], return_inverse=True)
    return lab


def scott_m(Xn):
    N, d = Xn.shape
    h = 3.5 * Xn.std(axis=0).mean() * N ** (-1.0 / (d + 2))
    return int(np.clip(round(1.0 / h), M_LADDER[0], M_LADDER[-1]))


def stable_m(Xn, k, offset, z, floor_count, return_scores=False):
    N, d = Xn.shape
    ladder = [m for m in M_LADDER if (m * k + 2 * k + 4) ** d <= 3e7]
    labs = [floptics_sa_sig(Xn, m, k, offset, z, floor_count) for m in ladder]
    ncls = [len(set(lb.tolist()) - {-1}) for lb in labs]
    # a single cluster is trivially "stable"; only consider it if nothing finer exists
    allow_one = max(ncls) <= 1
    scores = []
    for i in range(len(ladder)):
        nb = [j for j in (i - 1, i + 1) if 0 <= j < len(ladder)]
        s = np.mean([adjusted_rand_score(labs[i], labs[j]) for j in nb])
        if ncls[i] == 0 or (ncls[i] == 1 and not allow_one):
            s = -1.0
        scores.append(s)
    scores = np.array(scores)
    # ties: prefer the finer grid (more detail) among equally stable choices
    best = int(np.flatnonzero(scores >= scores.max() - 1e-9)[-1])
    if return_scores:
        return ladder[best], labs[best], dict(zip(ladder, np.round(scores, 3)))
    return ladder[best], labs[best]


def plateau_m(Xn, k, offset, z, floor_count, return_info=False):
    """Pick m from the longest run of consecutive ladder values that give the same number
    of clusters K >= 2 (the structure that survives the widest range of resolutions).
    Ties go to the finer run; inside the run the middle m is used. No labels needed."""
    N, d = Xn.shape
    ladder = [m for m in M_LADDER if (m * k + 2 * k + 4) ** d <= 3e7]
    labs = [floptics_sa_sig(Xn, m, k, offset, z, floor_count) for m in ladder]
    ks = [len(set(lb.tolist()) - {-1}) for lb in labs]
    runs, i = [], 0
    while i < len(ladder):
        j = i
        while j + 1 < len(ladder) and ks[j + 1] == ks[i]:
            j += 1
        runs.append((i, j, ks[i]))
        i = j + 1
    good = [r for r in runs if r[2] >= 2] or runs
    longest = max(r[1] - r[0] for r in good)
    i0, j0, _ = [r for r in good if r[1] - r[0] == longest][-1]
    pick = (i0 + j0 + 1) // 2
    if return_info:
        return ladder[pick], labs[pick], dict(zip(ladder, ks))
    return ladder[pick], labs[pick]


def stable_m_fixed(Xn, k, z, floor_count, n_internal=3):
    """Choose m WITHOUT looking at the user's grid offset: the stability score of every m
    is averaged over n_internal fixed offsets inside one fine cell (0, 1/(n k), ...).
    Because the averaged density repeats every 1/k of a bin, these offsets cover the only
    offsets that can matter. The chosen m depends on the data alone.
    Ladder: values with at most one coarse bin per point (m^d <= N) and a fine grid that fits
    in memory. Only interior ladder values are eligible, so every candidate is compared with
    a coarser AND a finer neighbour (end values, with one neighbour, look falsely stable)."""
    N, d = Xn.shape
    ladder = [m for m in M_LADDER if (m * k + 2 * k + 4) ** d <= 3e7 and m ** d <= N]
    if len(ladder) < 3:
        ladder = list(M_LADDER[:3])
    total = np.zeros(len(ladder))
    for t in range(n_internal):
        off = np.full(d, t / (n_internal * k))
        labs = [floptics_sa_sig(Xn, m, k, off, z, floor_count) for m in ladder]
        ncls = [len(set(lb.tolist()) - {-1}) for lb in labs]
        allow_one = max(ncls) <= 1
        for i in range(len(ladder)):
            if i == 0 or i == len(ladder) - 1:
                total[i] = -np.inf
                continue
            sc = np.mean([adjusted_rand_score(labs[i], labs[j]) for j in (i - 1, i + 1)])
            if ncls[i] == 0 or (ncls[i] == 1 and not allow_one):
                sc = -1.0
            total[i] += sc / n_internal
    best = int(np.flatnonzero(total >= total.max() - 1e-9)[-1])
    return ladder[best]


def floptics_auto(Xn, offset, k=4, z=1.0, floor_count=1.0, m_rule="stable_fixed",
                  n_sub=20000, rescale=True):
    """Fully automatic FLOPTICS-SA. Returns (labels, chosen m)."""
    if m_rule == "scott":
        m = scott_m(Xn)
        return floptics_sa_sig(Xn, m, k, offset, z, floor_count), m
    if m_rule == "stable_fixed":
        N, d = Xn.shape
        if n_sub and N > n_sub:
            # choose m on a fixed random subsample (same rows whatever the offset), then
            # rescale it to the full N with the ASH/KDE bandwidth rate h ~ N^(-1/(d+4))
            idx = np.random.RandomState(12345).choice(N, n_sub, replace=False)
            m_sub = stable_m_fixed(Xn[idx], k, z, floor_count)
            m = m_sub if not rescale else int(round(m_sub * (N / n_sub) ** (1.0 / (d + 4))))
            m = int(min(m, max(mm for mm in M_LADDER if (mm * k + 2 * k + 4) ** d <= 3e7)))
        else:
            m = stable_m_fixed(Xn, k, z, floor_count)
        return floptics_sa_sig(Xn, m, k, offset, z, floor_count), m
    if m_rule == "stable":
        m, lab = stable_m(Xn, k, offset, z, floor_count)
        return lab, m
    m, lab = plateau_m(Xn, k, offset, z, floor_count)
    return lab, m
