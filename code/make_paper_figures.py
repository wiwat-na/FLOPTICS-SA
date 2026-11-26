#!/usr/bin/env python3
"""Figures for the FLOPTICS-SA manuscript (vector PDF). Run from the code/ folder; reads ../results, writes ../paper_figures."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pilot_floptics import floptics, paper_dataset, scores  # noqa: E402
from pilot_grid_offset import normalise  # noqa: E402
from auto_core import floptics_auto, floptics_sa_sig  # noqa: E402

warnings.filterwarnings("ignore")
RES = Path(__file__).resolve().parent.parent / "results"
OUT = RES.parent / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "font.family": "serif", "mathtext.fontset": "cm",
                     "axes.spines.top": False, "axes.spines.right": False})
C_FL, C_K1, C_SA = "#d95f02", "#7570b3", "#1b9e77"
facts = {}


def scatter(ax, X, lab, title):
    noise = lab == -1
    ax.scatter(X[noise, 0], X[noise, 1], s=1.2, c="#cccccc", rasterized=True)
    ax.scatter(X[~noise, 0], X[~noise, 1], s=1.2, c=lab[~noise] % 8, cmap="Dark2",
               vmin=0, vmax=7, rasterized=True)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(True)


# ---------------------------------------------------------------- Fig. 1 + Fig. 2: dataset D
X, y = paper_dataset("D", 2026)
Xn = normalise(X)
rng = np.random.RandomState(7)
offs = rng.rand(30, 2)
offs[0] = 0
fl = [scores(y, floptics(Xn, 100, 10, s)) for s in offs]
acc = np.array([f["paper_accuracy"] for f in fl])
ari = np.array([f["ari"] for f in fl])
ib, iw = int(ari.argmax()), int(ari.argmin())
lab_sa_w, m_sa = floptics_auto(Xn, offs[iw])
sa_all = np.array([scores(y, floptics_auto(Xn, s)[0])["ari"] for s in offs])
facts["D_floptics_ari_range"] = [float(ari.min()), float(ari.max())]
facts["D_floptics_acc_range"] = [float(acc.min()), float(acc.max())]
facts["D_sa_ari_range"] = [float(sa_all.min()), float(sa_all.max())]
facts["D_sa_m"] = int(m_sa)

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9))
scatter(axes[0], Xn, floptics(Xn, 100, 10, offs[ib]),
        f"(a) FLOPTICS, best offset\nARI = {ari[ib]:.2f}")
scatter(axes[1], Xn, floptics(Xn, 100, 10, offs[iw]),
        f"(b) FLOPTICS, worst offset\nARI = {ari[iw]:.2f}")
scatter(axes[2], Xn, lab_sa_w,
        f"(c) FLOPTICS-SA, same offset as (b)\nARI = {scores(y, lab_sa_w)['ari']:.2f}")
fig.tight_layout()
fig.subplots_adjust(top=0.84)
fig.savefig(OUT / "fig1_offset_D.pdf", dpi=300)
plt.close(fig)

ts = np.linspace(0, 1, 40, endpoint=False)
sw = {"fl": [], "k1": [], "sa": []}
for t in ts:
    s = np.array([t, t])
    sw["fl"].append(scores(y, floptics(Xn, 100, 10, s))["ari"])
    sw["k1"].append(scores(y, floptics_sa_sig(Xn, m_sa, 1, s, 1.0, 1.0))["ari"])
    sw["sa"].append(scores(y, floptics_sa_sig(Xn, m_sa, 4, s, 1.0, 1.0))["ari"])
facts["sweep_ranges"] = {k: [float(min(v)), float(max(v))] for k, v in sw.items()}
fig, ax = plt.subplots(figsize=(4.8, 2.6))
ax.plot(ts, sw["fl"], "-o", ms=2.5, c=C_FL, label="FLOPTICS (Bin_size 100)")
ax.plot(ts, sw["k1"], "-s", ms=2.5, c=C_K1, label=f"FLOPTICS-SA, k = 1 (m = {m_sa})")
ax.plot(ts, sw["sa"], "-^", ms=2.5, c=C_SA, label=f"FLOPTICS-SA, k = 4 (m = {m_sa})")
for x in (0.25, 0.5, 0.75):
    ax.axvline(x, c="#999999", lw=0.5, ls=":")
ax.set_xlabel("grid offset t (fraction of one bin, along the diagonal)")
ax.set_ylabel("ARI")
ax.set_ylim(-0.05, 1.0)
ax.legend(loc="lower left", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig3_sweep_D.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. 4: locked test
lk = pd.read_csv(RES / "locked_summary.csv")
order = ["floptics_auto", "kde_auto", "floptics_auto_k1", "dpc", "hdbscan_1pct", "automato",
         "dbscan_knee", "gmm_bic", "walther_kde", "hdbscan_default", "floptics"]
names = {"floptics_auto": "FLOPTICS-SA", "kde_auto": "Gauss. KDE\n+ auto m",
         "floptics_auto_k1": "SA, k = 1", "dpc": "DPC", "hdbscan_1pct": "HDBSCAN\n(1%)",
         "automato": "AuToMATo", "dbscan_knee": "DBSCAN\n(knee)", "gmm_bic": "GMM\n(BIC)",
         "walther_kde": "Walther\nKDE", "hdbscan_default": "HDBSCAN\n(default)",
         "floptics": "FLOPTICS"}
tab = lk.pivot(index="dataset", columns="method", values="ari_mean")[order]
fig, ax = plt.subplots(figsize=(7.0, 2.9))
bp = ax.boxplot([tab[m].values for m in order], patch_artist=True, widths=0.6,
                medianprops=dict(color="black"), flierprops=dict(ms=2))
for i, b in enumerate(bp["boxes"]):
    b.set_facecolor(C_SA if i == 0 else "#dddddd")
ax.plot(range(1, len(order) + 1), [tab[m].mean() for m in order], "D", c="black", ms=3,
        label="mean")
ax.set_xticks(range(1, len(order) + 1))
ax.set_xticklabels([names[m] for m in order], fontsize=7)
ax.set_ylabel("ARI (38 locked test datasets)")
ax.legend(loc="upper right", frameon=False, bbox_to_anchor=(1.0, 1.12))
fig.tight_layout()
fig.savefig(OUT / "fig4_locked_boxplot.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. 5: DLBCL stability
st = pd.read_csv(RES / "dlbcl_stability_fig5.csv")
piv = st.pivot(index="sample", columns="method", values="stability")
cols = [("floptics_b20", "FLOPTICS (Bin_size 20)", C_FL, "o"),
        ("floptics_b50", "FLOPTICS (Bin_size 50)", "#e6ab02", "v"),
        ("floptics_auto_k1", "FLOPTICS-SA, k = 1", C_K1, "s"),
        ("floptics_auto", "FLOPTICS-SA, k = 4", C_SA, "^")]
facts["dlbcl_median_stability"] = {c: float(piv[c].median()) for c, *_ in cols}
fig, ax = plt.subplots(figsize=(7.0, 2.6))
xs = np.arange(len(piv))
for j, (c, lab_, colr, mk) in enumerate(cols):
    ax.plot(xs + (j - 1.5) * 0.12, piv[c].values, mk, ms=3, c=colr, label=lab_)
ax.set_xticks(xs)
ax.set_xticklabels(piv.index, fontsize=6, rotation=90)
ax.set_xlabel("FlowCAP-I DLBCL sample")
ax.set_ylabel("stability (mean pairwise ARI\nover 10 grid offsets)")
ax.set_ylim(0.25, 1.02)
ax.legend(loc="lower left", ncol=2, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig5_dlbcl_stability.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. 6: scaling
sc = pd.read_csv(RES / "scaling_final_all.csv")
sty = {"floptics_auto": ("FLOPTICS-SA (automatic)", C_SA, "^", 2.2),
       "floptics_sa_fixed": ("FLOPTICS-SA (m known)", "#66c2a5", "^", 1.2),
       "floptics_b20": ("FLOPTICS (Bin_size 20)", "#e41a1c", "P", 1.2),
       "floptics_b50": ("FLOPTICS (Bin_size 50)", "#377eb8", "p", 1.2),
       "dbscan_knee": ("DBSCAN (knee)", "#7570b3", "o", 1.2),
       "hdbscan_1pct": ("HDBSCAN (1%)", "#e7298a", "s", 1.2),
       "gmm_bic": ("GMM (BIC)", "#d95f02", "D", 1.2),
       "automato": ("AuToMATo", "#a6761d", "v", 1.2),
       "optics_xi": ("OPTICS", "#444444", "X", 1.2)}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=True)
for ax, d in zip(axes, (2, 3)):
    for mt, (lab_, colr, mk, lw) in sty.items():
        sd = sc[(sc.d == d) & (sc.method == mt)].sort_values("n")
        s = sd[sd.status == "ok"]
        if len(s):
            ax.loglog(s.n, s.seconds, "-" + mk, c=colr, lw=lw, ms=4.5, label=lab_)
        to = sd[sd.status.str.startswith("timeout")]
        if len(to):  # first size that hit the limit: open marker on the limit line
            ax.plot(to.n.iloc[0], 300, mk, mfc="none", mec=colr, ms=6, mew=1.0)
            if len(s):
                ax.plot([s.n.iloc[-1], to.n.iloc[0]], [s.seconds.iloc[-1], 300], ":",
                        c=colr, lw=0.9)
    ax.axhline(300, c="#999999", lw=0.6, ls="--")
    ax.text(1.05e4, 380, "300 s time limit", fontsize=7, color="#666666")
    ax.set_ylim(1e-3, 1.2e3)
    ax.set_xlabel("number of points N")
    ax.set_title(f"d = {d}")
    ax.grid(alpha=0.25, which="major")
axes[0].set_ylabel("run time (s)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7)
fig.tight_layout(rect=(0, 0.17, 1, 1))
fig.savefig(OUT / "fig6_scaling.pdf")
plt.close(fig)

(RES / "figure_facts.json").write_text(json.dumps(facts, indent=2))
print(json.dumps(facts, indent=2))
