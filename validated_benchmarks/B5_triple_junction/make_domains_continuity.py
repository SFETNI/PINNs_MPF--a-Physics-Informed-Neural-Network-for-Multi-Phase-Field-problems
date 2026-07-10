#!/usr/bin/env python
"""
"Four worker networks, one continuous field" — B5 triple junction (128x128).

A reader-facing animation that shows the 2x2 spatial decomposition (four worker
networks) AND that the four subdomains form one field with strong space continuity.
The real phase-field reference is split into four quadrants that periodically slide
apart (revealing four distinct sub-domains) and reassemble (grain boundaries reconnect
unbroken across the seams), while the field relaxes in time.

Complements the plain relaxation GIF; the 128 relaxation animation is untouched.

Data:  reference/b5_ref_90_to_120_128.npz
Usage: python make_domains_continuity.py   # -> media/triple_junction_domains_continuity.gif
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ  = os.path.join(HERE, "reference", "b5_ref_90_to_120_128.npz")
OUT  = os.path.join(HERE, "media", "triple_junction_domains_continuity.gif")

# ---------------------------------------------------------------- palette
BG, INK, SUB, FAINT = "#ffffff", "#1c232b", "#586472", "#9aa4af"
ACCENT = "#ec6f0b"
PH = ["#4472c4", "#e8820e", "#c62d2d", "#3d9e2c"]
CMAP = ListedColormap(PH)
# worker colors (one per subdomain): W1 NW, W2 NE, W3 SW, W4 SE
WC = {1: "#2f6db0", 2: "#2f9e6b", 3: "#8a5cc4", 4: "#e07b1a"}
TAU_C = 87.890625

# ---------------------------------------------------------------- field data
d = np.load(NPZ)
phi = d["phi"].astype(np.float32)                 # (82,4,128,128)
times = d["times"].astype(float)
NT = len(times)
# display labels at native frames (.T + origin lower matches the other B5 figures)
DISP = [phi[i].argmax(0).T for i in range(NT)]     # each 128x128
n = DISP[0].shape[0]; h = n // 2                    # 64

def field_at(tf):
    """interpolated display label map at physical time tf (smooth evolution)."""
    x = np.clip(tf / times[-1], 0, 1) * (NT - 1)
    i = int(np.floor(x)); a = x - i
    if i >= NT - 1:
        return DISP[-1]
    if a < 1e-3:
        return DISP[i]
    pj = (1 - a) * phi[i] + a * phi[i + 1]
    return pj.argmax(0).T

# quadrants: (row-half, col-half) in the display array; W-number; explode sign
QUAD = [
    ("TL", slice(h, n), slice(0, h), 1, (-1, +1)),   # top-left  = W1
    ("TR", slice(h, n), slice(h, n), 2, (+1, +1)),   # top-right = W2
    ("BL", slice(0, h), slice(0, h), 3, (-1, -1)),   # bot-left  = W3
    ("BR", slice(0, h), slice(h, n), 4, (+1, -1)),   # bot-right = W4
]
# base extent of each quadrant in axis coords [0,1]^2 (origin lower):
#   rows (y): bottom half -> [0,.5], top half -> [.5,1]; cols (x) similar
def base_extent(rs, cs):
    x0 = 0.0 if cs.start == 0 else 0.5
    y0 = 0.0 if rs.start == 0 else 0.5
    return [x0, x0 + 0.5, y0, y0 + 0.5]

# ---------------------------------------------------------------- easing
def ss(t):
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    return t * t * (3 - 2 * t)

# ---------------------------------------------------------------- timeline
FA, FB, FBH, FC, FD = 42, 16, 26, 20, 40      # evolve / explode / hold / reassemble / evolve
NF = FA + FB + FBH + FC + FD
FPS = 15
GAP_MAX = 0.11

def state(f):
    """returns (tf, gap, badge, pulse, caption)."""
    if f < FA:                                   # A: together, evolve 0 -> ~110
        p = f / FA
        return 110 * p, 0.0, ss((p - 0.15) / 0.3), 0.0, "one continuous field"
    f -= FA
    if f < FB:                                   # B: explode (field held at 110)
        return 110, GAP_MAX * ss(f / FB), 1.0, 0.0, "four worker networks"
    f -= FB
    if f < FBH:                                  # hold apart
        return 110, GAP_MAX, 1.0, 0.0, "one master PINN coordinates them"
    f -= FBH
    if f < FC:                                   # C: reassemble + continuity pulse
        g = GAP_MAX * (1 - ss(f / FC))
        return 110, g, 1.0, ss(1 - abs(f / FC - 0.6) / 0.4), "continuous across interfaces"
    f -= FC
    p = f / FD                                   # D: together, evolve 110 -> 200
    return 110 + 90 * p, 0.0, ss((0.2 - p) / 0.2), 0.0, "one continuous field"

# ---------------------------------------------------------------- figure
FW, FH = 6.6, 7.0
fig = plt.figure(figsize=(FW, FH), dpi=110)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0.06, 0.06, 0.88, 0.80])
ax.set_xlim(-0.24, 1.24); ax.set_ylim(-0.24, 1.24); ax.set_aspect("equal")
ax.set_axis_off()

def net_glyph(cx, cy, color, s=0.026, alpha=1.0):
    layers = [2, 3, 2]
    xs = np.linspace(-s, s, 3)
    pts = [[(cx + xs[li], cy + yy) for yy in np.linspace(-s, s, nn)]
           for li, nn in enumerate(layers)]
    for li in range(2):
        for a in pts[li]:
            for b in pts[li + 1]:
                ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=0.7,
                        alpha=0.55 * alpha, zorder=8)
    allp = [p for L in pts for p in L]
    ax.scatter([p[0] for p in allp], [p[1] for p in allp], s=11, color="white",
               edgecolors=color, linewidths=1.0, alpha=alpha, zorder=9)

def pill(cx, cy, text, fc, tc, ec, alpha, w, h, fs, z=9):
    """a rounded label centred at (cx, cy)."""
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.004,rounding_size=0.03",
                 linewidth=1.3, edgecolor=ec, facecolor=fc,
                 alpha=alpha, zorder=z))
    ax.text(cx, cy, text, color=tc, fontsize=fs, fontweight="bold",
            ha="center", va="center", alpha=alpha, zorder=z + 1)

def draw(f):
    ax.cla()
    ax.set_xlim(-0.24, 1.24); ax.set_ylim(-0.24, 1.24); ax.set_aspect("equal")
    ax.set_axis_off()
    tf, gap, badge, pulse, cap = state(f)
    D = field_at(tf)

    # continuity background: one seamless full-domain field beneath the tiles.
    # It stays fully opaque until the blocks are clearly apart, so the closed /
    # reassembled state is pixel-perfect continuous (no sliver can show through);
    # it fades out only once the tiles have visibly peeled away.
    lo, hi = 0.15 * GAP_MAX, 0.45 * GAP_MAX
    fullalpha = 1.0 - ss((gap - lo) / (hi - lo))
    if fullalpha > 0.01:
        ax.imshow(D, cmap=CMAP, vmin=0, vmax=3, origin="lower",
                  extent=[0, 1, 0, 1], interpolation="nearest",
                  alpha=fullalpha, zorder=2)

    for name, rs, cs, w, (sx, sy) in QUAD:
        sub = D[rs, cs]
        x0, x1, y0, y1 = base_extent(rs, cs)
        dx, dy = sx * gap, sy * gap
        ax.imshow(sub, cmap=CMAP, vmin=0, vmax=3, origin="lower",
                  extent=[x0 + dx, x1 + dx, y0 + dy, y1 + dy],
                  interpolation="nearest", zorder=3)
        # worker-colored border - fades out completely as the tiles close, so the
        # reassembled field reads as one continuous whole (no residual seam lines).
        bfade = ss(gap / GAP_MAX)
        if bfade > 0.01:
            lw = 1.5 + 3.0 * bfade
            ax.add_patch(plt.Rectangle((x0 + dx, y0 + dy), 0.5, 0.5, fill=False,
                                       edgecolor=WC[w], lw=lw, alpha=bfade, zorder=6))
        # "Worker N" label centred just outside each tile's outer edge, and a
        # small network glyph inside the tile corner; both fade in as the block
        # separates from the whole.
        wa = ss(gap / (0.45 * GAP_MAX))
        if wa > 0.02:
            lx = (x0 + x1) / 2 + dx
            ly = (y1 + dy + 0.055) if sy > 0 else (y0 + dy - 0.055)
            pill(lx, ly, "Worker %d" % w, WC[w], "white", "white", wa,
                 0.205, 0.058, 9.5, z=9)
            gx = (x0 + dx) + (0.42 if sx > 0 else 0.08)
            gy = (y0 + dy) + (0.42 if sy > 0 else 0.08)
            net_glyph(gx, gy, WC[w], alpha=0.6 * wa)

    # master PINN: one shared objective + one optimizer train all four workers
    # jointly (with overlap-halo continuity) -- the modern analogue of the legacy
    # master/worker sync. Drawn as a central hub linked to every worker while the
    # blocks are apart, with a token travelling worker -> master (coordination).
    ha = ss((gap - 0.6 * GAP_MAX) / (0.35 * GAP_MAX))
    if ha > 0.02:
        inner = {1: (0.5 - gap, 0.5 + gap), 2: (0.5 + gap, 0.5 + gap),
                 3: (0.5 - gap, 0.5 - gap), 4: (0.5 + gap, 0.5 - gap)}
        fr = (f % 12) / 12.0
        for wk, (ix, iy) in inner.items():
            ax.plot([0.5, ix], [0.5, iy], color=SUB, lw=1.5,
                    alpha=0.55 * ha, zorder=6, solid_capstyle="round")
            ax.scatter([ix], [iy], s=16, color=WC[wk], alpha=0.8 * ha, zorder=7)
            ax.scatter([ix + (0.5 - ix) * fr], [iy + (0.5 - iy) * fr], s=20,
                       color=WC[wk], alpha=0.85 * ha, edgecolors="white",
                       linewidths=0.5, zorder=7)
        pill(0.5, 0.5, "master PINN", BG, INK, INK, ha, 0.215, 0.066, 8.5, z=10)

    # continuity pulse: a soft glow that blooms along the closed seams and then
    # dissolves, signalling the grain boundaries reconnect unbroken across them.
    if pulse > 0.02 and gap < 0.03:
        ax.plot([0.5, 0.5], [0, 1], color=ACCENT, lw=12, alpha=0.22 * pulse,
                zorder=4, solid_capstyle="round")
        ax.plot([0, 1], [0.5, 0.5], color=ACCENT, lw=12, alpha=0.22 * pulse,
                zorder=4, solid_capstyle="round")

    # header + caption + time
    fig.texts.clear()
    fig.text(0.06, 0.955, "Four worker networks, one continuous field",
             fontsize=13.5, fontweight="bold", color=INK)
    fig.text(0.06, 0.917, "B5 triple junction  ·  2×2 spatial decomposition",
             fontsize=10.5, color=SUB)
    ccol = ACCENT if cap == "continuous across interfaces" else INK
    fig.text(0.5, 0.028, cap, fontsize=12.5, fontweight="bold", color=ccol,
             ha="center")
    fig.text(0.94, 0.917, f"t = {tf:5.1f}", fontsize=10, color=FAINT, ha="right")
    return []

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "x":
        draw(int(sys.argv[2]))
        fig.savefig(os.path.join(HERE, "media", f"_dom_test_{sys.argv[2]}.png"),
                    dpi=110, facecolor=BG)
        print("wrote test frame", "of", NF)
    else:
        anim = FuncAnimation(fig, draw, frames=NF, interval=1000/FPS, blit=False)
        anim.save(OUT, writer=PillowWriter(fps=FPS), dpi=110,
                  savefig_kwargs={"facecolor": BG})
        print("wrote", OUT, "frames=", NF)
