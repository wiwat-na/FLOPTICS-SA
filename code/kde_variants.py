"""
Gaussian-KDE variants of FLOPTICS-SA, used as baselines for the question
"what does the Averaged Shifted Histogram (ASH) add over a binned Gaussian KDE?"

Both variants keep EVERYTHING else of FLOPTICS-SA: Chebyshev adjacency, the significance
merge p - l <= z*sqrt(p + l) with z = 1, floor = 1, 0.5% minimum cluster size, and the
symmetric tie-break key. Only the density on the grid changes.

  walther_kde : the density of Walther et al. (2009): linear binning onto a grid that spans
                the data range (256 nodes per axis in 2D, 64 in 3D), product Gaussian kernel,
                bandwidth h_j = SD_j * n^(-1/6). No parameter is chosen from the data
                beyond this rule. (Walther's own pointer-chain merging and 4.3-sigma
                background test are NOT used; the merge is ours, so only the density differs.)
  kde_auto    : linear binning onto the same fine grid as FLOPTICS-SA (m*k nodes per axis),
                Gaussian kernel with the same standard deviation as the ASH triangle filter
                (sigma = sqrt((k^2 - 1)/6) fine cells), and m chosen by the same automatic
                rule. This isolates the kernel shape (triangle via shifted grids vs Gaussian).

Units. The merge test needs a density whose variance is about its mean (Poisson-like).
For S(g) = sum_i K_h(x_i - g): E S = n f(g), Var S ~ n f(g) R(K_h), R(K_h) = int K_h^2.
So D = S / R(K_h) has Var D ~ E D. For a product Gaussian, R(K_h) = prod 1/(2 sqrt(pi) h_j);
with grid spacing Delta and sigma_j = h_j / Delta (grid units), D = G * prod(2 sqrt(pi) sigma_j),
where G is the Gaussian-filtered linear-binned grid (filter weights sum to 1).
"""

import numpy as np
from scipy import ndimage
from sklearn.metrics import adjusted_rand_score

from auto_core import M_LADDER, _significance_merge
from pilot_invariant_floptics import chebyshev_offsets


def linear_bin(Xn, scale, off, pad, values=None):
    """Spread each point (weight 1, or `values`) over the 2^d grid nodes around it.
    Node i sits at coordinate (i - pad - off)/scale. Returns grid and each point's
    nearest node (flat index)."""
    N, d = Xn.shape
    u = Xn * scale + off + pad
    i0 = np.floor(u).astype(np.int64)
    fr = u - i0
    size = int(np.ceil(scale + 1 + np.max(off))) + 2 * pad + 2
    shape = (size,) * d
    G = np.zeros(int(np.prod(shape)))
    w_pt = np.ones(N) if values is None else values
    for corner in np.ndindex(*([2] * d)):
        corner = np.array(corner)
        w = np.prod(np.where(corner == 1, fr, 1 - fr), axis=1) * w_pt
        idx = np.ravel_multi_index((i0 + corner).T, shape)
        G += np.bincount(idx, weights=w, minlength=G.size)
    nearest = np.ravel_multi_index(np.rint(u).astype(np.int64).T, shape)
    return G.reshape(shape), nearest


def merge_and_label(D, key2, nearest, N, z, floor, min_frac):
    d = D.ndim
    D = D.copy()
    for ax in range(d):  # zero frame so neighbour lookups stay inside the array
        sl = [slice(None)] * d
        sl[ax] = 0
        D[tuple(sl)] = 0
        sl[ax] = -1
        D[tuple(sl)] = 0
    flat = D.ravel()
    k2 = key2.ravel()
    cand = np.flatnonzero(flat >= floor)
    order = cand[np.lexsort((-k2[cand], -flat[cand]))]
    parent = _significance_merge(flat, order, chebyshev_offsets(D.shape), z, k2)
    rp = parent[nearest]
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


def kde_on_grid(Xn, scale, off, sigma, z, floor, min_frac):
    N, d = Xn.shape
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (d,))
    pad = int(np.ceil(4 * sigma.max())) + 2
    G, nearest = linear_bin(Xn, scale, off, pad)
    v = ((Xn - 0.5 * (1 - 1e-9)) ** 2).sum(axis=1)
    Gv, _ = linear_bin(Xn, scale, off, pad, values=v)
    factor = np.prod(2 * np.sqrt(np.pi) * sigma)
    D = ndimage.gaussian_filter(G, sigma, truncate=4.0, mode="constant") * factor
    K2 = ndimage.gaussian_filter(Gv, sigma, truncate=4.0, mode="constant") * factor
    return merge_and_label(D, K2, nearest, N, z, floor, min_frac)


def walther_kde(Xn, offset, z=1.0, floor=1.0, min_frac=0.005):
    N, d = Xn.shape
    M = 256 if d == 2 else 64
    delta = 1.0 / (M - 1)
    h = Xn.std(axis=0) * N ** (-1.0 / 6.0)
    return kde_on_grid(Xn, M - 1, np.asarray(offset), h / delta, z, floor, min_frac)


def kde_sig(Xn, m, k, offset, z=1.0, floor=1.0, min_frac=0.005):
    sigma = np.sqrt((k * k - 1) / 6.0)
    return kde_on_grid(Xn, m * k, np.asarray(offset) * k, sigma, z, floor, min_frac)


def _stable_m(Xn, k, z, floor, n_internal=3):
    N, d = Xn.shape
    ladder = [m for m in M_LADDER if (m * k + 2 * k + 12) ** d <= 3e7 and m ** d <= N]
    if len(ladder) < 3:
        ladder = list(M_LADDER[:3])
    total = np.zeros(len(ladder))
    for t in range(n_internal):
        off = np.full(d, t / (n_internal * k))
        labs = [kde_sig(Xn, m, k, off, z, floor) for m in ladder]
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
    return ladder[int(np.flatnonzero(total >= total.max() - 1e-9)[-1])]


def kde_auto(Xn, offset, k=4, z=1.0, floor=1.0, n_sub=20000):
    N, d = Xn.shape
    if N > n_sub:
        idx = np.random.RandomState(12345).choice(N, n_sub, replace=False)
        m = int(round(_stable_m(Xn[idx], k, z, floor) * (N / n_sub) ** (1.0 / (d + 4))))
        m = int(min(m, max(mm for mm in M_LADDER if (mm * k + 2 * k + 12) ** d <= 3e7)))
    else:
        m = _stable_m(Xn, k, z, floor)
    return kde_sig(Xn, m, k, offset, z, floor), m
