# FLOPTICS-SA

Code and results for *"FLOPTICS-SA: Offset-Robust Grid Density Clustering with Shift-Averaged Histograms and Count-Based Peak Merging"* (W. Sriphum, M. Obthong; submitted to Pattern Recognition).

## Quick use
```python
import numpy as np
from auto_core import floptics_auto          # code/ folder on the Python path
from pilot_grid_offset import normalise
X = normalise(np.random.rand(10000, 2))     # any 2D or 3D data, rescaled to [0,1)^d
labels, m = floptics_auto(X, offset=np.full(2, 1/8))  # -1 = noise; s = 1/(2k), k = 4
```

## Reproducing the paper
Install (Python 3.11 or later): `pip install -r requirements.txt`. Benchmark data: clone https://github.com/gagolews/clustering-data-v1.

| Paper item | Command (run inside `code/`) |
|---|---|
| Table 1, Fig. S1 (locked test) | `python pilot_locked_test.py --data-dir <clustering-data-v1>` |
| Table 1 (development, test-1) | `python pilot_auto_floptics.py --data-dir <clustering-data-v1>` |
| Fig. 4 (DLBCL) | `python pilot_dlbcl.py --fcs-dir <DLBCL fcs folder> --cytometree <DLBCL.RData>` |
| Fig. 5, Table S2 (run time) | `python pilot_scaling_auto.py` (all methods; results merged in `results/scaling_final_all.csv`) |
| Statistical tests on the locked list (Friedman, Wilcoxon-Holm, Nemenyi) | `python stats/locked_stats.py` (post hoc, on stored locked results) |
| Equivariance check on locked data (fixed vs automatic m) | `python stats/check_equivariance_auto.py <clustering-data-v1>` |
| Audit of Theorem 2 (Supplementary S4): locked data at s=0, s=1/(2k), random offsets; DLBCL | `python stats/audit_thm2.py <clustering-data-v1>`, `python stats/audit_thm2_random.py <clustering-data-v1>`, `python stats/audit_thm2_dlbcl.py <DLBCL fcs folder>` |
| Misty-Mountain-like comparison (Supplementary S6) | `python stats/misty_like.py <clustering-data-v1>` |
| Eq. (3) vs ToMATo on sqrt(D), whole runs (Supplementary S6) | `python stats/tomato_sandwich.py <clustering-data-v1>`, `python stats/tomato_nesting.py <clustering-data-v1>` |
| Audit of Proposition 2 and the merge rule (Supplementary S5) | `python stats/audit_prop2_variance.py`, `python stats/audit_merge_rule.py`, `python stats/z_sensitivity.py <clustering-data-v1>` |
| DPC without its N > 8,000 memory guard, sensitivity of the locked comparison (Supplementary S3) | `python stats/dpc_large_n.py <clustering-data-v1>` |
| Audit of the automatic choice of m (Supplementary S7) | `python stats/audit_m_oracle.py <clustering-data-v1>`, `python stats/audit_m_synthetic.py`, `python stats/audit_m_noise_rescale.py` |
| Figs. 1, 3-5 and S1 | `python make_paper_figures.py` (reads `results/`, writes `paper_figures/`) |

`results/LOCK.json` holds the SHA-256 hashes of the code files and the settings, written before the locked test was run (2026-11-25 06:27 UTC). The eight code files listed in it are unchanged since then (`sha256sum code/*.py` to check); `pilot_scaling_auto.py` and `make_paper_figures.py` are not part of the locked test and were updated later.

FlowCAP-I DLBCL: FlowRepository FR-FCM-ZZYY. Labels of sample 003: cytometree R package (data/DLBCL.RData).

## License
MIT (see `LICENSE`).

## Note on AI assistance
Code and experiments were developed with the assistance of Claude (Anthropic); see Section "Implementation and use of AI-assisted tools" of the paper.
