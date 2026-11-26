#!/usr/bin/env python3
"""
LOCKED final test on datasets that were never used while FLOPTICS-SA was designed.

Protocol
  1. The method and every setting are frozen before this script runs. The SHA-256 hashes of
     the code files and the settings are written to results_locked/LOCK.json BEFORE any
     dataset is clustered, so anyone can check that nothing changed afterwards.
  2. The test datasets below were chosen by one rule, before looking at any result:
     every 2D dataset of the clustering-data-v1 suite with at most 10,000 points that was
     not in the development or the earlier test lists. (No 3D dataset meets the rule.)
  3. Each method runs once per dataset (grid methods: 10 random grid offsets).
  4. The numbers printed here are the ones to report, whatever they are.

Usage
  python pilot_locked_test.py --data-dir clustering-data-v1
"""

import argparse
import hashlib
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pilot_auto_floptics import load, run_method
from pilot_floptics import scores
from pilot_grid_offset import normalise

warnings.filterwarnings("ignore")

LOCKED_TEST = [
    "cdc/DS1", "cdc/DS4", "cdc/DS6", "cdc/DS7", "cdc/DS8", "cdc/DS10", "cdc/DS11",
    "cdc/DS12", "cdc/DS13", "cdc/DS15", "cdc/DS16", "cdc/DS17",
    "g2mg/g2mg_2_10", "g2mg/g2mg_2_20", "g2mg/g2mg_2_30", "g2mg/g2mg_2_40", "g2mg/g2mg_2_50",
    "g2mg/g2mg_2_60", "g2mg/g2mg_2_70", "g2mg/g2mg_2_80", "g2mg/g2mg_2_90",
    "h2mg/h2mg_2_10", "h2mg/h2mg_2_20", "h2mg/h2mg_2_30", "h2mg/h2mg_2_40", "h2mg/h2mg_2_50",
    "h2mg/h2mg_2_60", "h2mg/h2mg_2_70", "h2mg/h2mg_2_80", "h2mg/h2mg_2_90",
    "other/chameleon_t4_8k", "other/chameleon_t5_8k", "other/chameleon_t7_10k",
    "other/chameleon_t8_8k", "sipu/a3", "sipu/unbalance", "wut/isolation", "wut/trajectories",
]
METHODS = ["floptics_auto", "floptics_auto_k1", "kde_auto", "walther_kde", "floptics",
           "hdbscan_default", "hdbscan_1pct", "automato", "gmm_bic", "dbscan_knee", "dpc"]
DETERMINISTIC = {"hdbscan_default", "hdbscan_1pct", "automato", "gmm_bic", "dbscan_knee", "dpc"}
CODE = ["auto_core.py", "kde_variants.py", "baselines_free.py", "pilot_auto_floptics.py",
        "pilot_floptics.py", "pilot_grid_offset.py", "pilot_invariant_floptics.py",
        "pilot_locked_test.py"]
SETTINGS = {"floptics_auto": {"k": 4, "z": 1.0, "floor": 1.0, "min_frac": 0.005,
                              "m_rule": "stable_fixed", "n_sub": 20000, "rescale": True},
            "runs_per_grid_method": 10, "seed": 2026}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", default="results_locked")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    lock = {"created_utc": datetime.now(timezone.utc).isoformat(),
            "datasets": LOCKED_TEST, "methods": METHODS, "settings": SETTINGS,
            "sha256": {f: hashlib.sha256((here / f).read_bytes()).hexdigest() for f in CODE}}
    (out / "LOCK.json").write_text(json.dumps(lock, indent=2))
    print("LOCK written:", out / "LOCK.json")

    rng = np.random.RandomState(SETTINGS["seed"])
    rows = []
    for nm in LOCKED_TEST:
        X, y = load(nm, args.data_dir, SETTINGS["seed"])
        Xn = normalise(X)
        offs = rng.rand(SETTINGS["runs_per_grid_method"], Xn.shape[1])
        t1 = time.time()
        for method in METHODS:
            n_runs = 1 if method in DETERMINISTIC else len(offs)
            for r in range(n_runs):
                t = time.perf_counter()
                lab, m = run_method(method, Xn, offs[r])
                rows.append(dict(dataset=nm, n=len(y), method=method, run=r, m=m,
                                 seconds=time.perf_counter() - t, **scores(y, lab)))
        print(f"  {nm:24s} n={len(y):5d} {time.time() - t1:6.1f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "locked_runs.csv", index=False)
    agg = (df.groupby(["dataset", "method"])
             .agg(ari_mean=("ari", "mean"), ari_std=("ari", "std"), sec=("seconds", "mean"))
             .reset_index())
    agg.to_csv(out / "locked_summary.csv", index=False)
    tab = agg.pivot(index="dataset", columns="method", values="ari_mean")[METHODS]
    std = agg.pivot(index="dataset", columns="method", values="ari_std")[METHODS]
    sec = agg.pivot(index="dataset", columns="method", values="sec")[METHODS]
    pd.set_option("display.width", 250)
    print("\n=== LOCKED TEST: mean ARI per dataset ===")
    print(tab.round(3).to_string())
    print("\nmedian ARI : " + "  ".join(f"{m}={tab[m].median():.3f}" for m in METHODS))
    print("mean ARI   : " + "  ".join(f"{m}={tab[m].mean():.3f}" for m in METHODS))
    print("median std : " + "  ".join(f"{m}={std[m].median():.3f}" for m in METHODS
                                     if m not in DETERMINISTIC))
    print("median sec : " + "  ".join(f"{m}={sec[m].median():.3f}" for m in METHODS))
    for other in METHODS[1:]:
        dl = tab["floptics_auto"] - tab[other]
        print(f"floptics_auto vs {other:16s}: higher {int((dl > 0.01).sum()):2d}  "
              f"lower {int((dl < -0.01).sum()):2d}  tie {int((dl.abs() <= 0.01).sum()):2d}")
    # mean rank (1 = best) across datasets
    ranks = tab.rank(axis=1, ascending=False).mean().sort_values()
    print("\nmean rank (1 = best): " + "  ".join(f"{m}={v:.2f}" for m, v in ranks.items()))


if __name__ == "__main__":
    main()
