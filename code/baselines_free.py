"""
Label-free baselines: each one uses a published default or rule of thumb, never the
reference labels.

  optics_xi    OPTICS (scikit-learn), min_samples=5, automatic xi extraction (xi=0.05)
  gmm_bic      Gaussian mixture, full covariances, K = 1..15 chosen by BIC
  dbscan_knee  DBSCAN, min_samples = 2*d (Ester et al. 1996 / Sander et al. 1998 rule),
               eps = knee of the sorted k-distance curve (point furthest from the chord)
  dpc          Density peaks (Rodriguez & Laio, Science 2014): cutoff d_c = 2% quantile of
               pairwise distances, Gaussian-kernel density; centres = points whose
               gamma = rho*delta lies above mean + 3*std of gamma (a common automatic rule).
               Skipped when N > 8000 (needs the full distance matrix).
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


def optics_xi(X):
    return OPTICS(min_samples=5).fit_predict(X)


def gmm_bic(X, kmax=15, seed=0):
    best, best_bic = None, np.inf
    for k in range(1, kmax + 1):
        g = GaussianMixture(n_components=k, covariance_type="full", random_state=seed,
                            reg_covar=1e-6).fit(X)
        b = g.bic(X)
        if b < best_bic:
            best, best_bic = g, b
    return best.predict(X)


def knee_eps(X, k):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, _ = nn.kneighbors(X)
    kd = np.sort(dist[:, -1])
    n = len(kd)
    x = np.linspace(0, 1, n)
    y = (kd - kd[0]) / max(kd[-1] - kd[0], 1e-12)
    # distance of each point of the curve to the straight line from first to last point
    knee = int(np.argmax(np.abs(y - x)))
    return float(kd[knee])


def dbscan_knee(X):
    ms = 2 * X.shape[1]
    eps = knee_eps(X, ms - 1)
    return DBSCAN(eps=eps, min_samples=ms).fit_predict(X)


def dpc(X, max_n=8000):
    n = len(X)
    if n > max_n:
        return None
    D = squareform(pdist(X)).astype(np.float32)
    dc = np.quantile(D[np.triu_indices(n, 1)], 0.02)
    rho = np.exp(-(D / dc) ** 2).sum(axis=1) - 1
    order = np.argsort(-rho)
    delta = np.zeros(n)
    nearest = np.zeros(n, dtype=int)
    delta[order[0]] = D[order[0]].max()
    nearest[order[0]] = order[0]
    for i in range(1, n):
        p = order[i]
        higher = order[:i]
        j = higher[np.argmin(D[p, higher])]
        delta[p] = D[p, j]
        nearest[p] = j
    gamma = rho * delta
    centres = np.flatnonzero(gamma > gamma.mean() + 3 * gamma.std())
    if order[0] not in centres:
        centres = np.append(centres, order[0])
    lab = np.full(n, -1)
    lab[centres] = np.arange(len(centres))
    for p in order:  # assign in decreasing density, each point follows its nearest denser point
        if lab[p] == -1:
            lab[p] = lab[nearest[p]]
    return lab
