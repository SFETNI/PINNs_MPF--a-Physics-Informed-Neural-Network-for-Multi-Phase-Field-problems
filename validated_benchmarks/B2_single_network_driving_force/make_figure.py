#!/usr/bin/env python
"""B2 - "one network is not enough" figure (driven-force grain shrinkage).

Reader-facing motivation figure. It is NOT a validated benchmark: it demonstrates
WHY PINNs-MPF decomposes the domain. On one identical driven-force time window
(Delta_g = -250), a single global network cannot hold the sharp interface -- the
grain amplitude collapses (peak phi ~ 0.55, a smooth blob) -- whereas the 2x2
four-worker decomposition restores the interface (peak phi ~ 0.95). Both are
compared against the same finite-difference reference (peak phi = 1.0).

All numbers are read from the two metrics.json files in data/ so the figure stays
faithful to the recorded runs; nothing is hand-entered.

Usage: python make_figure.py   # -> figures/b2_single_vs_four_network.png
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
D1 = json.load(open(os.path.join(HERE, "data", "single_network_metrics.json")))
D4 = json.load(open(os.path.join(HERE, "data", "four_worker_metrics.json")))

# --- pull the recorded scalars (no hand-entered numbers) --------------------
def peak_phi(d):
    return float(d["diag_per_interval"][0]["pinn"]["max_phi"])
def ref_peak(d):
    return float(d["diag_per_interval"][0]["ref"]["max_phi"])
def mse(d):
    return float(d["mse_vs_reference_mean"])

p1, p4, pref = peak_phi(D1), peak_phi(D4), ref_peak(D1)
m1, m4 = mse(D1), mse(D4)
dg = D1["delta_g"]
assert D1["boxes"] == 1 and D4["boxes"] == 4

# ---------------------------------------------------------------- style
BG, INK, SUB, FAINT = "#ffffff", "#1c232b", "#586472", "#9aa4af"
RED, GREEN, GRAY = "#c0392b", "#2f9e6b", "#8895a4"
GRAIN = "#c62d2d"

fig = plt.figure(figsize=(11.2, 5.0), dpi=100)
fig.patch.set_facecolor(BG)

# =============================================================== left: schematic
axL = fig.add_axes([0.035, 0.10, 0.40, 0.74])
axL.set_xlim(0, 1); axL.set_ylim(0, 1); axL.set_aspect("equal"); axL.set_axis_off()

def domain(cx, cy, s, split, peak, label, ok):
    """draw a domain box; grain opacity encodes the recorded peak phi (amplitude)."""
    x0, y0 = cx - s / 2, cy - s / 2
    # the grain: a disc whose fill strength = peak phi (visual amplitude cue)
    axL.add_patch(Circle((cx, cy), s * 0.30, facecolor=GRAIN, alpha=float(peak),
                         edgecolor="none", zorder=3))
    axL.add_patch(Circle((cx, cy), s * 0.30, facecolor="none",
                         edgecolor=GRAIN, lw=1.2, alpha=0.35, zorder=3))
    # domain frame / decomposition
    if split:
        for k in (0, 1, 2):
            axL.plot([x0 + k * s / 2] * 2, [y0, y0 + s], color=SUB, lw=1.3, zorder=4)
            axL.plot([x0, x0 + s], [y0 + k * s / 2] * 2, color=SUB, lw=1.3, zorder=4)
        for gx in (x0 + s * 0.25, x0 + s * 0.75):
            for gy in (y0 + s * 0.25, y0 + s * 0.75):
                axL.scatter([gx], [gy], s=90, color="white", edgecolors=GREEN,
                            linewidths=1.6, zorder=6)
    else:
        axL.add_patch(plt.Rectangle((x0, y0), s, s, fill=False, edgecolor=SUB,
                                    lw=1.6, zorder=4))
        axL.scatter([cx], [cy + s * 0.02], s=120, color="white", edgecolors=RED,
                    linewidths=1.8, zorder=6)
    axL.text(cx, y0 - 0.05, label, ha="center", va="top", fontsize=11.5,
             fontweight="bold", color=INK)
    axL.text(cx, y0 - 0.115, ("holds the interface" if ok else "interface collapses"),
             ha="center", va="top", fontsize=9.5, fontweight="bold",
             color=(GREEN if ok else RED))

domain(0.26, 0.60, 0.42, split=False, peak=p1, label="single network", ok=False)
domain(0.74, 0.60, 0.42, split=True,  peak=p4, label="four workers (2×2)", ok=True)
axL.text(0.5, 0.985, "same domain, same driven window", ha="center", va="top",
         fontsize=10, color=SUB)

# =============================================================== right: bars
axP = fig.add_axes([0.52, 0.55, 0.44, 0.31])   # peak phi
axM = fig.add_axes([0.52, 0.11, 0.44, 0.28])   # mse

# peak phi bars (amplitude / interface sharpness) -- the honest common scale
names = ["single\nnetwork", "four\nworkers", "reference"]
vals = [p1, p4, pref]
cols = [RED, GREEN, GRAY]
bars = axP.bar(names, vals, color=cols, width=0.62, zorder=3)
axP.axhline(1.0, ls="--", lw=1.2, color=FAINT, zorder=2)
axP.text(-0.55, 1.10, r"- - - sharp-interface target ($\phi$ = 1)", va="center",
         ha="left", fontsize=8.5, color=SUB)
for b, v in zip(bars, vals):
    axP.text(b.get_x() + b.get_width() / 2, v + 0.03, "%.2f" % v, ha="center",
             va="bottom", fontsize=10.5, fontweight="bold",
             color=b.get_facecolor())
axP.set_ylim(0, 1.18); axP.set_yticks([0, 0.5, 1.0])
axP.set_ylabel(r"peak $\phi$", fontsize=10.5)
axP.set_title(r"interface amplitude at the end of the window", fontsize=10.5,
              color=INK, pad=6)
for sp in ("top", "right"):
    axP.spines[sp].set_visible(False)
axP.set_xlim(-0.6, 2.6)

# mse bars (lower is better)
mnames = ["single\nnetwork", "four\nworkers"]
mvals = [m1, m4]
mbars = axM.bar(mnames, mvals, color=[RED, GREEN], width=0.46, zorder=3)
for b, v in zip(mbars, mvals):
    axM.text(b.get_x() + b.get_width() / 2, v + 0.004, "%.3f" % v, ha="center",
             va="bottom", fontsize=10.5, fontweight="bold",
             color=b.get_facecolor())
axM.set_ylim(0, max(mvals) * 1.25)
axM.set_ylabel("MSE vs reference", fontsize=10.5)
axM.set_title("field error (lower is better)", fontsize=10.5, color=INK, pad=6)
for sp in ("top", "right"):
    axM.spines[sp].set_visible(False)
axM.set_xlim(-0.6, 1.6)

# =============================================================== titles
fig.text(0.035, 0.955,
         "Why PINNs-MPF decomposes the domain: one network is not enough",
         fontsize=15, fontweight="bold", color=INK)
fig.text(0.035, 0.905,
         r"Driven-force grain shrinkage  ·  $\Delta g = %g$  ·  identical time window  ·  "
         r"reference peak $\phi$ = 1.0" % dg,
         fontsize=10.5, color=SUB)

OUT = os.path.join(HERE, "figures", "b2_single_vs_four_network.png")
fig.savefig(OUT, dpi=100, facecolor=BG)
print("wrote", OUT)
print("peak phi: single=%.3f four=%.3f ref=%.3f | mse single=%.3f four=%.3f"
      % (p1, p4, pref, m1, m4))
