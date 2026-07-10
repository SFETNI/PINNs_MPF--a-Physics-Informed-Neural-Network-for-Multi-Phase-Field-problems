#!/usr/bin/env python
"""
Static "method overview" infographic for PINNs-MPF (Figure B).

A clean, horizontal five-stage pipeline of the softmax MultiNN workflow, anchored to a
REAL benchmark example: the B5 triple junction relaxing from a 90-degree initial
condition toward the 120-degree Young-angle network (4 phases, 2x2 spatial
decomposition, 8 time windows).  The microstructure thumbnails (initial field, stitched
output, FD reference, |difference|) are cropped from the benchmark's own validation
figure -- not generic art -- and the metrics are the run's actual values.  Grid size is
deliberately not printed: the decomposition is illustrative and scales to any N x N.

Source figure (real data):
  validated_benchmarks/B5_triple_junction/figures/
      b5_triple_junction_reference_vs_pinns_mpf_128.png
Reference field (for the interface-weighted sampling panel):
  validated_benchmarks/B5_triple_junction/reference/b5_ref_90_to_120_128.npz

Usage:  python make_workflow_figure.py            # -> pinns_mpf_workflow.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
B5 = os.path.join(HERE, "..", "validated_benchmarks", "B5_triple_junction")
SRC_FIG = os.path.join(B5, "figures",
                       "b5_triple_junction_reference_vs_pinns_mpf_128.png")
REF_NPZ = os.path.join(B5, "reference", "b5_ref_90_to_120_128.npz")

# ----------------------------------------------------------------- palette
BG      = "#ffffff"
INK     = "#1c232b"
SUBINK  = "#586472"
FAINT   = "#8b96a3"
CARD    = "#f6f8fb"
CARDED  = "#dbe2ea"
BADGE   = "#2f6db0"
ACCENT  = "#ec6f0b"
ACC_D   = "#c85c05"
# sampling point categories / loss terms
C_PDE   = "#3b7dd8"
C_IC    = "#3f9e57"
C_CONT  = "#e8720c"
C_SEAM  = "#8a5cc4"
C_DN    = "#8a94a0"
# 4-phase colors (match the benchmark figure legend)
PH = ["#4472c4", "#e8820e", "#c62d2d", "#3d9e2c"]   # phase 1..4

# ----------------------------------------------------------------- real thumbnails
def crop_square(box):
    im = Image.open(SRC_FIG).convert("RGB")
    return np.asarray(im.crop(box).resize((256, 256), Image.LANCZOS))

# 128-run figure: rows FD (181,508) / PINNs-MPF (538,865) / |diff| (896,1222);
# columns t=0 (56,383) ... t=150 (1558,1885).  The t=0 FD panel is the 90-degree IC;
# t=150 panels are clean (no junction-marker overlay).
THUMBS = {
    "ic":   crop_square((65,   190,  374,  499)),   # FD reference, t=0   -> 90-degree IC
    "out":  crop_square((1567, 547, 1876,  856)),   # PINNs-MPF,   t=150  (stitched output)
    "ref":  crop_square((1567, 190, 1876,  499)),   # FD reference, t=150
    "diff": crop_square((1569, 907, 1876, 1211)),   # |difference|, t=150
}

# interface-weighted sampling: label map at t~50 (curved junction network)
_ref = np.load(REF_NPZ)
_lab = _ref["phi"][20].argmax(0)                     # (128,128) argmax phase index

# ----------------------------------------------------------------- figure scaffold
FW, FH = 15.6, 8.7
fig = plt.figure(figsize=(FW, FH), dpi=120)
fig.patch.set_facecolor(BG)
bg = fig.add_axes([0, 0, 1, 1]); bg.set_axis_off()
bg.set_xlim(0, 1); bg.set_ylim(0, 1)

MARGIN, GAP, NC = 0.016, 0.011, 5
CW = (1 - 2*MARGIN - (NC-1)*GAP) / NC
CY0, CY1 = 0.090, 0.850
def cx0(i): return MARGIN + i*(CW+GAP)

def card(i, title, badge):
    x0 = cx0(i)
    box = FancyBboxPatch((x0, CY0), CW, CY1-CY0,
                         boxstyle="round,pad=0.006,rounding_size=0.012",
                         linewidth=1.1, edgecolor=CARDED, facecolor=CARD,
                         mutation_aspect=FW/FH, zorder=1)
    bg.add_patch(box)
    # number badge (scatter renders a true circle regardless of axis aspect)
    bx, by = x0 + 0.026, CY1 - 0.030
    bg.scatter([bx], [by], s=560, color=BADGE, zorder=4, edgecolors="white",
               linewidths=1.5)
    bg.text(bx, by, badge, color="white", fontsize=14, fontweight="bold",
            ha="center", va="center", zorder=5)
    bg.text(bx + 0.028, by, title, color=INK, fontsize=12.5, fontweight="bold",
            ha="left", va="center", zorder=5)
    return x0

def txt(x, y, s, size=9.2, color=SUBINK, weight="normal", ha="left", va="center",
        style="normal"):
    bg.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
            style=style, zorder=6)

def dot(x, y, color, s=70):
    bg.scatter([x], [y], s=s, color=color, zorder=6, edgecolors="white",
               linewidths=0.8)

def arrow_between(i):
    """small connector arrow in the gap after card i."""
    x = cx0(i) + CW + GAP*0.5
    y = (CY0 + CY1) / 2
    bg.annotate("", xy=(x+0.010, y), xytext=(x-0.010, y),
                arrowprops=dict(arrowstyle="-|>", color=BADGE, lw=2.4), zorder=3)

def thumb_axes(cx, cy, w, h):
    """add an inset axes in figure coords for a raster thumbnail."""
    ax = fig.add_axes([cx, cy, w, h]); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#c4ccd6"); s.set_linewidth(1.0)
    return ax

def phase_legend(x, y):
    labs = ["φ₁", "φ₂", "φ₃", "φ₄"]
    for k in range(4):
        dot(x + k*0.026, y, PH[k], s=60)
        bg.text(x + k*0.026, y-0.024, labs[k], fontsize=7.6, color=SUBINK,
                ha="center", va="center", zorder=6)

def net_glyph(ax, cx, cy, color, sc=1.0):
    layers = [2, 3, 2]
    xs = np.linspace(-0.16*sc, 0.16*sc, len(layers))
    pts = []
    for li, n in enumerate(layers):
        ys = np.linspace(-0.16*sc, 0.16*sc, n)
        pts.append([(cx+xs[li], cy+yy) for yy in ys])
    for li in range(len(layers)-1):
        for a in pts[li]:
            for b in pts[li+1]:
                ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=0.7,
                        alpha=0.6, zorder=3)
    allp = [p for L in pts for p in L]
    ax.scatter([p[0] for p in allp], [p[1] for p in allp], s=16, color="white",
               edgecolors=color, linewidths=1.1, zorder=4)

# ----------------------------------------------------------------- header
bg.text(MARGIN, 0.955, "PINNs-MPF", fontsize=20, fontweight="bold", color=INK)
bg.text(MARGIN, 0.917, "softmax MultiNN workflow", fontsize=12.5, color=SUBINK)
bg.text(1-MARGIN, 0.945, "Example benchmark", fontsize=10, color=FAINT, ha="right")
bg.text(1-MARGIN, 0.918, "B5 triple junction · 90° → 120° relaxation",
        fontsize=11, color=ACC_D, ha="right", fontweight="bold")

for i in range(NC-1):
    arrow_between(i)

# ================================================================= CARD 1
x0 = card(0, "Window setup", "1")
cxc = x0 + CW/2
txt(x0+0.014, 0.788, "current time window", size=9.0, color=SUBINK)
# timeline t_{k-1} -- t_k -- t_{k+1}
ty = 0.755
bg.plot([x0+0.022, x0+CW-0.022], [ty, ty], color=FAINT, lw=1.4, zorder=3)
for frac, lab, big in [(0.0, "$t_{k-1}$", False), (0.5, "$t_k$", True),
                       (1.0, "$t_{k+1}$", True)]:
    px = x0+0.022 + frac*(CW-0.044)
    dot(px, ty, BADGE if big else FAINT, s=90 if big else 45)
    bg.text(px, ty-0.026, lab, fontsize=8.6, color=INK if big else FAINT,
            ha="center", va="center", zorder=6)
bg.annotate("", xy=(x0+0.022+0.55*(CW-0.044), ty+0.016),
            xytext=(x0+0.022+0.45*(CW-0.044), ty+0.016),
            arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.6), zorder=6)
# initial field thumbnail
txt(cxc, 0.692, "initial field  /  handoff", size=9.4, color=INK, weight="bold",
    ha="center")
tw = 0.100; th = tw*FW/FH
ax1 = thumb_axes(cxc-tw/2, 0.472, tw, th)
ax1.imshow(THUMBS["ic"])
txt(cxc, 0.452, "multiphase field at $t_k$", size=8.4, color=SUBINK, ha="center")
phase_legend(cxc-0.039, 0.416)
txt(x0+0.016, 0.352, "Provided by a previous-window", size=8.2, color=SUBINK)
txt(x0+0.016, 0.330, "prediction, a reference handoff,", size=8.2, color=SUBINK)
txt(x0+0.016, 0.308, "or the analytic first condition.", size=8.2, color=SUBINK)
txt(x0+0.016, 0.266, "Next-window IC", size=8.6, color=ACC_D, weight="bold")
txt(x0+0.016, 0.244, "autonomous mode: predicted field", size=8.0, color=SUBINK)
txt(x0+0.016, 0.222, "B5 validation: FD start frame", size=8.0, color=SUBINK)

# ================================================================= CARD 2
x0 = card(1, "Sampling", "2")
txt(x0+0.014, 0.788, "points drawn in the window $[t_k, t_{k+1}]$", size=8.6)
# scatter panel — points are denser ALONG THE INTERFACES (grain boundaries) than
# inside the bulk grains, mirroring the interface-band collocation the solver uses.
sw = 0.115; sh = sw*FW/FH
sax = fig.add_axes([x0+CW/2-sw/2, 0.560, sw, sh]); sax.set_xticks([]); sax.set_yticks([])
for s in sax.spines.values(): s.set_edgecolor("#c4ccd6")
sax.set_xlim(0,1); sax.set_ylim(0,1)
rng = np.random.default_rng(7)
n = _lab.shape[0]
# interface mask from the real field (pixel differs from a 4-neighbour)
intf = np.zeros_like(_lab, bool)
intf[:-1,:] |= _lab[:-1,:] != _lab[1:,:]; intf[1:,:] |= _lab[:-1,:] != _lab[1:,:]
intf[:,:-1] |= _lab[:,:-1] != _lab[:,1:]; intf[:,1:] |= _lab[:,:-1] != _lab[:,1:]
iy, ix = np.where(intf)
# faint grain-boundary network so the reader sees WHY the points cluster
sax.scatter(ix/n, iy/n, s=0.8, color="#d9dee6", zorder=1)
# PDE collocation: dense on the interface (+ small jitter), sparse in the bulk
sel = rng.choice(len(ix), 46, replace=False)
jx = rng.uniform(-.012,.012,46); jy = rng.uniform(-.012,.012,46)
sax.scatter(ix[sel]/n+jx, iy[sel]/n+jy, s=8, color=C_PDE, zorder=3)
sax.scatter(rng.uniform(.05,.95,9), rng.uniform(.05,.95,9), s=6, color=C_PDE,
            alpha=0.75, zorder=2)
# IC (bottom strip) · internal continuity (2x2 seam) · periodic seam (edges)
sax.scatter(rng.uniform(.06,.94,9), rng.uniform(.02,.10,9), s=9, color=C_IC, zorder=4)
sax.scatter(np.full(6,.5)+rng.uniform(-.015,.015,6), rng.uniform(.1,.9,6), s=9,
            color=C_CONT, zorder=4)
sax.scatter(rng.choice([0.015,0.985],8), rng.uniform(.1,.9,8), s=9, color=C_SEAM, zorder=4)
sax.axvline(.5, color=C_CONT, lw=0.7, alpha=0.35)
cats = [(C_PDE, "PDE collocation", "dense along the interfaces"),
        (C_IC, "initial condition", "anchor to the handoff at $t_k$"),
        (C_CONT, "internal continuity", "match across subdomain borders"),
        (C_SEAM, "periodic seam", "wrap across domain boundaries"),
        (C_DN, "bulk stabilization", "sample stable bulk phases")]
yy = 0.470
for c, name, desc in cats:
    dot(x0+0.022, yy, c, s=64)
    txt(x0+0.040, yy, name, size=8.6, color=INK, weight="bold")
    txt(x0+0.040, yy-0.020, desc, size=7.5, color=SUBINK)
    yy -= 0.052

# ================================================================= CARD 3
x0 = card(2, "Spatial MultiNN", "3")
cxc = x0+CW/2
txt(x0+0.014, 0.790, "2×2 domain decomposition", size=9.0, color=INK, weight="bold")
txt(x0+0.014, 0.768, "one worker network per subdomain", size=8.4, color=SUBINK)
# 2x2 grid of workers
gw = 0.118; gh = gw*FW/FH
gax = fig.add_axes([cxc-gw/2, 0.500, gw, gh]); gax.set_xticks([]); gax.set_yticks([])
gax.set_xlim(0,1); gax.set_ylim(0,1)
for s in gax.spines.values(): s.set_edgecolor("#c4ccd6")
gax.axvline(.5, color="#c4ccd6", lw=1.0); gax.axhline(.5, color="#c4ccd6", lw=1.0)
cent = [(.25,.75,PH[0],"W1"),(.75,.75,PH[1],"W2"),(.25,.25,PH[2],"W3"),(.75,.25,PH[3],"W4")]
for cx,cy,col,lab in cent:
    gax.add_patch(Rectangle((cx-.24,cy-.24),.48,.48, facecolor=col, alpha=0.12, zorder=1))
    net_glyph(gax, cx, cy+0.04, col, sc=0.72)
    gax.text(cx, cy-0.22, lab, fontsize=7.6, color=col, ha="center", va="center",
             fontweight="bold", zorder=5)
txt(cxc, 0.462, "each worker → 4 logits  $z_1..z_4$", size=8.8, color=INK, ha="center",
    weight="bold")
txt(cxc, 0.438, "for any $(x, y, t)$ in its subdomain", size=8.0, color=SUBINK, ha="center")
# scalability note
sc_box = FancyBboxPatch((x0+0.014, 0.300), CW-0.028, 0.070,
                        boxstyle="round,pad=0.004,rounding_size=0.008",
                        linewidth=1.0, edgecolor="#cfe0f2", facecolor="#eef5fc",
                        mutation_aspect=FW/FH, zorder=2)
bg.add_patch(sc_box)
txt(x0+0.026, 0.348, "Scales to any  N×N", size=9.0, color=BADGE, weight="bold")
txt(x0+0.026, 0.326, "2×2, 3×3, 4×4, 6×6, … — chosen", size=7.8, color=SUBINK)
txt(x0+0.026, 0.308, "to match the problem complexity.", size=7.8, color=SUBINK)
txt(x0+0.016, 0.250, "Overlapping halos let neighbouring", size=7.8, color=SUBINK)
txt(x0+0.016, 0.230, "workers enforce continuity.", size=7.8, color=SUBINK)

# ================================================================= CARD 4
x0 = card(3, "Softmax + PI loss", "4")
cxc = x0+CW/2
txt(x0+0.014, 0.788, "per point $(x, y, t)$", size=9.0, color=SUBINK)
# logits -> softmax -> phi
txt(x0+0.022, 0.720, "$z_1 .. z_4$", size=12, color=INK, ha="left")
bg.annotate("", xy=(cxc+0.045, 0.720), xytext=(cxc-0.012, 0.720),
            arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.8), zorder=6)
txt(cxc+0.017, 0.742, "softmax", size=8.2, color=ACC_D, ha="center", weight="bold")
txt(x0+CW-0.022, 0.720, "$\\phi_1 .. \\phi_4$", size=12, color=INK, ha="right")
# constraint pill
pill = FancyBboxPatch((cxc-0.070, 0.640), 0.140, 0.048,
                      boxstyle="round,pad=0.004,rounding_size=0.02",
                      linewidth=1.0, edgecolor="#f3d3b0", facecolor="#fdf1e6",
                      mutation_aspect=FW/FH, zorder=2)
bg.add_patch(pill)
txt(cxc, 0.664, "$\\sum_i \\phi_i = 1,\\ \\ 0 \\leq \\phi_i \\leq 1$", size=9.4,
    color=ACC_D, ha="center", weight="bold")
txt(cxc, 0.612, "enforced by construction", size=7.8, color=SUBINK, ha="center",
    style="italic")
# loss box
lb = FancyBboxPatch((x0+0.014, 0.470), CW-0.028, 0.092,
                    boxstyle="round,pad=0.005,rounding_size=0.01",
                    linewidth=1.1, edgecolor="#cdd7e2", facecolor="#ffffff",
                    mutation_aspect=FW/FH, zorder=2)
bg.add_patch(lb)
txt(cxc, 0.522, r"$\mathcal{L}=\mathcal{L}_{pde}+\mathcal{L}_{ic}+"
    r"\mathcal{L}_{dn}+\mathcal{L}_{cont}+\mathcal{L}_{pbc}$",
    size=11.5, color=INK, ha="center", weight="bold")
terms = [(C_PDE, "$\\mathcal{L}_{pde}$", "PDE residual"),
         (C_IC, "$\\mathcal{L}_{ic}$", "initial condition"),
         (C_DN, "$\\mathcal{L}_{dn}$", "bulk stabilization"),
         (C_CONT, "$\\mathcal{L}_{cont}$", "internal continuity"),
         (C_SEAM, "$\\mathcal{L}_{pbc}$", "periodic seam")]
yy = 0.420
for c, sym, desc in terms:
    dot(x0+0.024, yy, c, s=58)
    txt(x0+0.040, yy, sym, size=9.6, color=INK)
    txt(x0+0.082, yy, desc, size=8.2, color=SUBINK)
    yy -= 0.040

# ================================================================= CARD 5
x0 = card(4, "Output + validation", "5")
cxc = x0+CW/2
txt(x0+0.014, 0.788, "assemble · validate · prepare next IC", size=9.0, color=SUBINK)
# stitched output thumbnail
txt(cxc, 0.742, "stitched field at $t_{k+1}$", size=9.2, color=INK, weight="bold",
    ha="center")
tw = 0.100; th = tw*FW/FH
oax = thumb_axes(cxc-tw/2, 0.545, tw, th); oax.imshow(THUMBS["out"])
# validation pair: reference | difference
txt(x0+0.016, 0.500, "compared against FD reference", size=8.4, color=SUBINK)
pw = 0.070; ph = pw*FW/FH
rax = thumb_axes(x0+0.026, 0.330, pw, ph); rax.imshow(THUMBS["ref"])
dax = thumb_axes(x0+0.026+pw+0.016, 0.330, pw, ph); dax.imshow(THUMBS["diff"])
txt(x0+0.026+pw/2, 0.318, "reference", size=7.8, color=SUBINK, ha="center")
txt(x0+0.026+pw+0.016+pw/2, 0.318, "|difference|", size=7.8, color=SUBINK, ha="center")
# metrics
mets = [("MSE vs reference", "3.7e-4  (t=200)"),
        ("argmax disagreement", "1.2 – 1.7%"),
        ("angle-tracking gap", "0.40°  (t=200)"),
        ("triple junctions", "6 / 6 stable"),
        ("phase-sum error", "≤ 4.4e-16")]
yy = 0.264
for name, val in mets:
    txt(x0+0.016, yy, name, size=8.0, color=SUBINK)
    txt(x0+CW-0.016, yy, val, size=8.2, color=INK, weight="bold", ha="right")
    yy -= 0.029

# ----------------------------------------------------------------- bottom banner
ban = FancyBboxPatch((MARGIN, 0.016), 1-2*MARGIN, 0.052,
                     boxstyle="round,pad=0.004,rounding_size=0.012",
                     linewidth=1.2, edgecolor=BADGE, facecolor="#f3f8fd",
                     mutation_aspect=FW/FH, zorder=2)
bg.add_patch(ban)
bg.text(0.5, 0.052, "4 spatial workers × 4 logits  →  softmax-constrained multiphase field",
        fontsize=12.5, color=INK, ha="center", va="center", fontweight="bold", zorder=6)
bg.text(0.5, 0.030, "scalable domain decomposition + physics-informed learning "
        "for accurate, stable multiphase evolution",
        fontsize=9.2, color=SUBINK, ha="center", va="center", zorder=6)

fig.savefig(os.path.join(HERE, "pinns_mpf_workflow.png"), dpi=120, facecolor=BG)
print("wrote pinns_mpf_workflow.png")
