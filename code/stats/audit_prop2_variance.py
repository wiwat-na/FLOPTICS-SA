"""Proposition 2 beyond its assumptions: Var D(j) vs E D(j) for fine counts that are
(a) Poisson with a common mean (the proposition), (b) Poisson with a linear gradient,
(c) multinomial (N fixed), (d) negative binomial with var/mean = 2 (overdispersed)."""
import numpy as np
from scipy import ndimage
rng = np.random.RandomState(0); k = 4; G = 64; R = 4000
w1 = (k - np.abs(np.arange(-(k - 1), k))) / k
def D(c):
    out = c
    for ax in range(c.ndim):
        out = ndimage.correlate1d(out, w1, axis=ax, mode="constant")
    return out
for d in (2, 3):
    shape = (G,) * d if d == 2 else (24,) * 3
    j = tuple(s // 2 for s in shape); a, b = j, tuple(x + k for x in j)
    lam = 20.0; mu0 = lam / k ** d
    grad = np.ones(shape) * mu0
    grad *= np.linspace(0.5, 1.5, shape[0]).reshape((-1,) + (1,) * (d - 1))
    e = np.zeros(shape); e[a] = 1; wa = D(e); e = np.zeros(shape); e[b] = 1; wb = D(e)  # weights as fields
    for nm_, mu in (("common", np.full(shape, mu0)), ("gradient", grad)):
        ex_v = (wa ** 2 * mu).sum() / (wa * mu).sum()
        ex_d = ((wa - wb) ** 2 * mu).sum() / ((wa + wb) * mu).sum()
        print(f"d={d} exact {nm_:8s} Var/E={ex_v:.3f}  diff ratio={ex_d:.3f}")
    res = {}
    for name in ("poisson_common", "poisson_gradient", "multinomial", "negbin_vm2"):
        Ds = []; Dd = []; Db = []
        for _ in range(R):
            if name == "poisson_common": c = rng.poisson(mu0, shape)
            elif name == "poisson_gradient": c = rng.poisson(grad)
            elif name == "multinomial":
                p = np.full(np.prod(shape), 1 / np.prod(shape))
                c = rng.multinomial(int(mu0 * np.prod(shape)), p).reshape(shape)
            else:  # NB with mean mu0, var 2*mu0: n = mu0, p = 0.5
                c = rng.negative_binomial(mu0, 0.5, shape)
            Dc = D(c.astype(float)); Ds.append(Dc[j]); Dd.append(Dc[a] - Dc[b]); Db.append(Dc[b])
        Ds = np.array(Ds); Dd = np.array(Dd); Db = np.array(Db)
        meanD = Ds.mean(); EDa_b = meanD + Db.mean()  # j = a, so E D(a) + E D(b)
        print(f"d={d} {name:17s} E D={meanD:6.2f}  Var D={Ds.var():6.2f}  Var/E={Ds.var()/meanD:5.3f}"
              f"  formula factor={((2*k*k+1)/(3*k*k))**d:5.3f}  Var(Da-Db)/(EDa+EDb)={Dd.var()/EDa_b:5.3f}")
