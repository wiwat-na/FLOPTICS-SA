"""Post hoc sensitivity of FLOPTICS-SA to z on the locked list (same datasets, loader and
offset generator as the locked run; only z changes)."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import adjusted_rand_score as ari
from pilot_locked_test import LOCKED_TEST, load, SETTINGS
from pilot_grid_offset import normalise
from auto_core import floptics_auto
rng = np.random.RandomState(SETTINGS["seed"]); rows = []
for nm in LOCKED_TEST:
    X, y = load(nm, sys.argv[1], SETTINGS["seed"]); Xn = normalise(X)
    offs = rng.rand(SETTINGS["runs_per_grid_method"], Xn.shape[1])
    for z in (0.5, 1.0, 1.5, 2.0, 3.0):
        a = [ari(y, floptics_auto(Xn, s, z=z)[0]) for s in offs]
        rows.append(dict(dataset=nm, z=z, ari=np.mean(a)))
    print(nm, flush=True)
R = pd.DataFrame(rows); R.to_csv("z_sensitivity_locked.csv", index=False)
print(R.groupby("z").ari.agg(["mean", "median"]).round(3).to_string())
