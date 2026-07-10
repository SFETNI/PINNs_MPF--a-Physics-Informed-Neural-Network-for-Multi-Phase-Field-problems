#!/usr/bin/env python
"""
Animated space-time decomposition schematic for PINNs-MPF (Figure A, animated).

A clean, technical, *representative* hero loop (not a literal per-benchmark trace)
that explains the method in ~14 s:
  - 2x2 spatial decomposition of the 2D domain (4 tiles / 4 worker networks)
  - extrusion into successive time windows W1..W8
  - an active window that marches W1 -> W8 as a solid slab
  - internal continuity + periodic seam + window-to-window handoff

Quality pass (v2): slower pacing with dwell/holds, eased transitions, the active
window rendered as a real 3D slab, a soft ground shadow, a marching progress rail,
refined palette + typography.  Representative labels are generic (W1..W8); they are
NOT meant to reproduce any single benchmark's exact window count.

Usage:  python make_spacetime_animation.py out.gif
        python make_spacetime_animation.py x 100      # dump a single test frame
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

# ----------------------------------------------------------------------------- palette
BG        = "#ffffff"
INK       = "#1c232b"
SUBINK    = "#5a6672"
FAINT     = "#9aa4af"
ACCENT    = "#ec6f0b"          # active window / progression accent
ACCENT_D  = "#c85c05"
SEAM      = "#1f6fd0"          # periodic seam
CONT      = "#2a2f36"          # internal continuity nodes
SHADOW    = "#c9d0d8"

# 4 refined, harmonious subdomain colors (NW, NE, SW, SE) = 4 worker networks
TILE = {
    "NW": "#5b8fd6",   # blue    (1)
    "NE": "#5aa981",   # teal    (2)
    "SW": "#9a77c9",   # violet  (3)
    "SE": "#e2a63f",   # amber   (4)
}
TILE_DARK = {"NW": "#33639f", "NE": "#337a56", "SW": "#6b4a9b", "SE": "#b97f1a"}
TILE_WALL = {"NW": "#3f6ba6", "NE": "#3d8360", "SW": "#74549f", "SE": "#c08b2b"}

NWIN = 8
DZ   = 1.05
MID  = 0.5
TH   = 0.16                     # slab thickness for the active window

TILES = {
    "NW": (0.0, MID, MID, 1.0),
    "NE": (MID, 1.0, MID, 1.0),
    "SW": (0.0, MID, 0.0, MID),
    "SE": (MID, 1.0, 0.0, MID),
}
TILE_NUM = {"NW": "1", "NE": "2", "SW": "3", "SE": "4"}

# ----------------------------------------------------------------------------- easing
def clamp01(t):
    return 0.0 if t < 0 else 1.0 if t > 1 else t

def ss(t):                       # smoothstep
    t = clamp01(t)
    return t*t*(3 - 2*t)

def seg(t, a, b):                # eased ramp active over [a,b]
    return ss((t - a) / max(1e-9, (b - a)))

# ----------------------------------------------------------------------------- geometry
def quad(x0, x1, y0, y1, z):
    return [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]

def rgba(c, a):
    return (*matplotlib.colors.to_rgb(c), a)

def tile_inset(k, gap):
    x0, x1, y0, y1 = TILES[k]
    ix0 = x0 + (gap if x0 == MID else 0.0)
    ix1 = x1 - (gap if x1 == MID else 0.0)
    iy0 = y0 + (gap if y0 == MID else 0.0)
    iy1 = y1 - (gap if y1 == MID else 0.0)
    return ix0, ix1, iy0, iy1

def add_plane(ax, z, alpha, edge_alpha=0.30, gap=0.012, dark=False, nums=False,
              numsize=13):
    """Flat 2x2 decomposition plane at height z."""
    polys, cols = [], []
    for k in TILES:
        ix0, ix1, iy0, iy1 = tile_inset(k, gap)
        polys.append(quad(ix0, ix1, iy0, iy1, z))
        cols.append(rgba(TILE_DARK[k] if dark else TILE[k], alpha))
    pc = Poly3DCollection(polys, facecolors=cols, linewidths=0.5,
                          edgecolors=rgba("#000000", edge_alpha))
    pc.set_sort_zpos(z)
    ax.add_collection3d(pc)
    if nums:
        for k in TILES:
            x0, x1, y0, y1 = TILES[k]
            ax.text((x0+x1)/2, (y0+y1)/2, z+0.015, TILE_NUM[k], color="white",
                    fontsize=numsize, fontweight="bold", ha="center", va="center",
                    zorder=20)

def add_slab(ax, z, alpha_top, th=TH, glow=1.0):
    """Active window as a solid extruded slab (top tiles + perimeter walls)."""
    z0, z1 = z - th/2, z + th/2
    # perimeter side walls (outer square only), colored per owning tiles
    walls = [
        # (face polygon, color)
        ([(0, 0, z0), (MID, 0, z0), (MID, 0, z1), (0, 0, z1)], TILE_WALL["SW"]),
        ([(MID, 0, z0), (1, 0, z0), (1, 0, z1), (MID, 0, z1)], TILE_WALL["SE"]),
        ([(0, 1, z0), (MID, 1, z0), (MID, 1, z1), (0, 1, z1)], TILE_WALL["NW"]),
        ([(MID, 1, z0), (1, 1, z0), (1, 1, z1), (MID, 1, z1)], TILE_WALL["NE"]),
        ([(0, 0, z0), (0, MID, z0), (0, MID, z1), (0, 0, z1)], TILE_WALL["SW"]),
        ([(0, MID, z0), (0, 1, z0), (0, 1, z1), (0, MID, z1)], TILE_WALL["NW"]),
        ([(1, 0, z0), (1, MID, z0), (1, MID, z1), (1, 0, z1)], TILE_WALL["SE"]),
        ([(1, MID, z0), (1, 1, z0), (1, 1, z1), (1, MID, z1)], TILE_WALL["NE"]),
    ]
    wp = Poly3DCollection([w[0] for w in walls],
                          facecolors=[rgba(w[1], 0.85) for w in walls],
                          edgecolors=rgba("#000000", 0.12), linewidths=0.4)
    wp.set_sort_zpos(z - th/2)
    ax.add_collection3d(wp)
    # top face tiles
    add_plane(ax, z1, alpha_top, edge_alpha=0.45, gap=0.012, nums=False)
    # accent cage: top + bottom outline + 4 vertical corner posts
    for zz in (z0, z1):
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [zz]*5, color=ACCENT,
                lw=2.6, alpha=glow, zorder=11)
    for (cx, cy) in [(0, 0), (1, 0), (1, 1), (0, 1)]:
        ax.plot([cx, cx], [cy, cy], [z0, z1], color=ACCENT, lw=2.6, alpha=glow,
                zorder=11)
    # soft glow ring above
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [z1+0.03]*5, color=ACCENT,
            lw=1.0, alpha=0.30*glow, zorder=11)
    return z1

def ground_shadow(ax, alpha):
    if alpha <= 0:
        return
    dx, dy, zz = 0.06, -0.05, -0.06
    poly = quad(0.02+dx, 1.02+dx, -0.02+dy, 0.98+dy, zz)
    pc = Poly3DCollection([poly], facecolors=[rgba(SHADOW, 0.5*alpha)],
                          edgecolors="none")
    pc.set_sort_zpos(zz-0.01)
    ax.add_collection3d(pc)

# ----------------------------------------------------------------------------- glyphs
def net_glyph(ax, cx, cy, z, color, scale=0.13, alpha=1.0, lw=0.9):
    layers = [2, 3, 3, 2]
    xs_layer = np.linspace(-scale, scale, len(layers))
    pts = []
    for li, n in enumerate(layers):
        ys = np.linspace(-scale*0.72, scale*0.72, n)
        pts.append([(cx + xs_layer[li], cy + yy) for yy in ys])
    segs = []
    for li in range(len(layers) - 1):
        for a in pts[li]:
            for b in pts[li+1]:
                segs.append([(a[0], a[1], z), (b[0], b[1], z)])
    lc = Line3DCollection(segs, colors=[rgba(color, alpha*0.65)], linewidths=lw)
    ax.add_collection3d(lc)
    allx = [p[0] for L in pts for p in L]
    ally = [p[1] for L in pts for p in L]
    ax.scatter(allx, ally, [z]*len(allx), s=24, color="white", alpha=alpha,
               depthshade=False, zorder=14, edgecolors=color, linewidths=1.3)

def continuity_marks(ax, z, alpha):
    if alpha <= 0:
        return
    ys = np.linspace(0.09, 0.91, 5)
    ax.plot([MID]*len(ys), ys, [z]*len(ys), color=CONT, lw=1.2, alpha=alpha*0.8)
    ax.scatter([MID]*len(ys), ys, [z]*len(ys), s=15, color="white",
               edgecolors=CONT, linewidths=1.0, alpha=alpha, depthshade=False,
               zorder=15)
    xs = np.linspace(0.09, 0.91, 5)
    ax.plot(xs, [MID]*len(xs), [z]*len(xs), color=CONT, lw=1.2, alpha=alpha*0.8)
    ax.scatter(xs, [MID]*len(xs), [z]*len(xs), s=15, color="white",
               edgecolors=CONT, linewidths=1.0, alpha=alpha, depthshade=False,
               zorder=15)

def seam_arcs(ax, z, alpha):
    if alpha <= 0:
        return
    t = np.linspace(0, 1, 22)
    # x-wrap along the front edge
    xa = t; ya = np.full_like(t, -0.02) - 0.07*np.sin(np.pi*t)
    za = np.full_like(t, z) + 0.11*np.sin(np.pi*t)
    ax.plot(xa, ya, za, color=SEAM, lw=2.0, alpha=alpha)
    ax.add_artist(Arrow3D([xa[-3], xa[-1]], [ya[-3], ya[-1]], [za[-3], za[-1]],
                          mutation_scale=10, lw=0, arrowstyle="-|>", color=SEAM,
                          alpha=alpha))
    ax.add_artist(Arrow3D([xa[2], xa[0]], [ya[2], ya[0]], [za[2], za[0]],
                          mutation_scale=10, lw=0, arrowstyle="-|>", color=SEAM,
                          alpha=alpha))
    # y-wrap along the left edge
    ya = t; xa = np.full_like(t, -0.02) - 0.07*np.sin(np.pi*t)
    za = np.full_like(t, z) + 0.11*np.sin(np.pi*t)
    ax.plot(xa, ya, za, color=SEAM, lw=2.0, alpha=alpha)
    ax.add_artist(Arrow3D([xa[-3], xa[-1]], [ya[-3], ya[-1]], [za[-3], za[-1]],
                          mutation_scale=10, lw=0, arrowstyle="-|>", color=SEAM,
                          alpha=alpha))

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *a, **k):
        super().__init__((0, 0), (0, 0), *a, **k)
        self._v = (xs, ys, zs)
    def do_3d_projection(self, renderer=None):
        xs, ys, zs = self._v
        x2, y2, z2 = proj3d.proj_transform(xs, ys, zs, self.axes.M)
        self.set_positions((x2[0], y2[0]), (x2[1], y2[1]))
        return float(np.min(z2))

def wlab(k):
    """Representative, general labels: W1..W7 concrete, top window is W_N (any N)."""
    return r"$W_{%d}$" % k if k < NWIN else r"$W_N$"

def win_label(ax, k, z, state):
    """state: 'future' | 'active' | 'done'."""
    if state == "active":
        col, fw, fs = ACCENT, "bold", 12
    elif state == "done":
        col, fw, fs = SUBINK, "normal", 9.5
    else:
        col, fw, fs = FAINT, "normal", 9.5
    ax.text(-0.17, 0.5, z, wlab(k), color=col, fontsize=fs, fontweight=fw,
            ha="center", va="center", zorder=12)
    # progress pip
    if state == "active":
        ax.scatter([-0.10], [0.5], [z], s=46, facecolor="white",
                   edgecolors=ACCENT, linewidths=2.2, depthshade=False, zorder=13)
    elif state == "done":
        ax.scatter([-0.10], [0.5], [z], s=34, color=ACCENT, depthshade=False,
                   zorder=13, alpha=0.9)
    else:
        ax.scatter([-0.10], [0.5], [z], s=26, color="#dfe4ea", depthshade=False,
                   zorder=13)

# ----------------------------------------------------------------------------- text
def txt(x, y, s, **kw):
    fig.text(x, y, s, **kw)

def title_block(sub=None, sub_accent=True):
    txt(0.035, 0.945, "PINNs-MPF", fontsize=18, fontweight="bold", color=INK)
    txt(0.035, 0.902, "space–time decomposition", fontsize=12.5, color=SUBINK)
    if sub:
        txt(0.035, 0.852, sub, fontsize=10.8,
            color=ACCENT if sub_accent else SUBINK, fontweight="bold")

def caption(s, strong=False):
    txt(0.5, 0.055, s, ha="center", fontsize=10.5 if strong else 10,
        color=INK if strong else SUBINK, fontweight="bold" if strong else "normal")

# A single FIXED legend, drawn identically every frame once tiles exist.  Keeping it
# static (no rows appearing/disappearing) is intentional: the earlier per-phase legend
# flickered.  Vertical spacing is generous to use the free right-hand column.
LEGEND_ROWS = [
    ("head",   None,        "4 worker networks",     False),
    ("swatch", TILE["NW"],  "tile 1  (NW)",          False),
    ("swatch", TILE["NE"],  "tile 2  (NE)",          False),
    ("swatch", TILE["SW"],  "tile 3  (SW)",          False),
    ("swatch", TILE["SE"],  "tile 4  (SE)",          False),
    ("gap",    None,        "",                       False),
    ("cage",   ACCENT,      "active window",         True),
    ("cont",   CONT,        "internal continuity",   False),
    ("seam",   SEAM,        "periodic seam (x, y)",  False),
]

def full_legend():
    x, y, dy = 0.795, 0.905, 0.056
    for kind, color, label, bold in LEGEND_ROWS:
        if kind == "head":
            txt(x, y, label, color=INK, fontsize=10.2, fontweight="bold", va="center")
            y -= 0.044
            continue
        if kind == "gap":
            y -= dy*0.55
            continue
        if kind == "swatch":
            txt(x, y, "■", color=color, fontsize=14, va="center")
        elif kind == "cage":
            txt(x, y, "□", color=color, fontsize=14, va="center", fontweight="bold")
        elif kind == "cont":
            txt(x, y, "─○─", color=color, fontsize=12, va="center")
        elif kind == "seam":
            txt(x, y, "↺", color=color, fontsize=15, va="center")
        txt(x + 0.038, y, label, color=ACCENT if bold else SUBINK, fontsize=9.6,
            va="center", fontweight="bold" if bold else "normal")
        y -= dy

# ----------------------------------------------------------------------------- axes
def base_axes(ax):
    ax.add_artist(Arrow3D([-0.06, -0.06], [-0.06, -0.06], [0.0, NWIN*DZ + 0.35],
                          mutation_scale=14, lw=2.0, arrowstyle="-|>", color=INK))
    ax.text(-0.06, -0.06, NWIN*DZ + 0.52, "Time", color=INK, fontsize=12,
            fontweight="bold", ha="center")
    ax.add_artist(Arrow3D([0, 1.16], [-0.03, -0.03], [0, 0],
                          mutation_scale=11, lw=1.5, arrowstyle="-|>", color=INK))
    ax.text(1.22, -0.05, 0, "x", color=INK, fontsize=11, fontweight="bold")
    ax.add_artist(Arrow3D([-0.03, -0.03], [0, 1.16], [0, 0],
                          mutation_scale=11, lw=1.5, arrowstyle="-|>", color=INK))
    ax.text(-0.05, 1.22, 0, "y", color=INK, fontsize=11, fontweight="bold")

# ----------------------------------------------------------------------------- timeline
F_INTRO   = 20        # domain fades in + hold
F_SPLIT   = 26        # split into 4 tiles + hold
F_EXTRUDE = 46        # windows stack up + hold
F_MARCH_W = 15        # frames per window during marching (dwell)
F_MARCH   = F_MARCH_W * NWIN
F_FINAL   = 26
NFRAMES   = F_INTRO + F_SPLIT + F_EXTRUDE + F_MARCH + F_FINAL
FPS       = 15        # slower playback

B1 = F_INTRO
B2 = B1 + F_SPLIT
B3 = B2 + F_EXTRUDE
B4 = B3 + F_MARCH
B5 = B4 + F_FINAL

fig = plt.figure(figsize=(9.6, 6.8), dpi=110)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection="3d")

# ---- independent inset: "scalable decomposition" (2x2 -> 4x4 -> 6x6 -> ... N x N).
# The 2x2 in the main scene is only the simplest case; the same MultiNN code runs any
# nx*ny (one worker per box).  This inset advertises that generality on a slow, held
# cycle so it never flickers.
INSET_BOX = [0.775, 0.155, 0.185, 0.205]
inset = fig.add_axes(INSET_BOX)
INSET_NS = [2, 4, 6]
INSET_PERIOD = 34                      # frames each grid is held (~2.3 s at 15 fps)

def draw_inset(show):
    inset.cla()
    inset.set_axis_off()
    if not show:
        inset.set_visible(False)
        return
    inset.set_visible(True)
    inset.set_xlim(-0.04, 1.04); inset.set_ylim(-0.04, 1.04)
    inset.set_aspect("equal")
    n = INSET_NS[(inset._frame // INSET_PERIOD) % len(INSET_NS)]
    cols = [TILE["NW"], TILE["NE"], TILE["SW"], TILE["SE"]]
    for i in range(n):
        for j in range(n):
            inset.add_patch(Rectangle((i/n, j/n), 1/n, 1/n,
                            facecolor=cols[(i + 2*j) % 4], edgecolor="white",
                            linewidth=0.9, alpha=0.85))
    inset.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor=INK, linewidth=1.6))
    inset.text(0.5, 1.17, "scalable decomposition", ha="center", va="bottom",
               fontsize=9.4, fontweight="bold", color=INK)
    inset.text(0.5, -0.14, r"$%d\times%d$" % (n, n), ha="center", va="top",
               fontsize=12, fontweight="bold", color=ACCENT_D)
    inset.text(0.5, -0.40, "any  N×N  — match the problem", ha="center", va="top",
               fontsize=7.6, color=SUBINK)
inset._frame = 0

def setup_axes(azim):
    ax.cla()
    ax.set_facecolor(BG)
    ax.set_axis_off()
    ax.set_xlim(-0.18, 1.32)
    ax.set_ylim(-0.18, 1.32)
    ax.set_zlim(-0.25, NWIN*DZ + 0.6)
    try:
        ax.set_box_aspect((1, 1, 1.4))
    except Exception:
        pass
    ax.view_init(elev=21, azim=azim)

# ----------------------------------------------------------------------------- frame
def draw(frame):
    fig.texts.clear()
    # Fixed camera: a stable schematic reads cleaner, and an unchanging background
    # lets GIF delta-optimization shrink the file dramatically (motion comes from
    # the content — extruding slabs, the marching active window — not the camera).
    setup_axes(-57)
    base_axes(ax)
    inset._frame = frame

    # ------------------------------------------------ INTRO: one domain
    if frame < B1:
        p = ss(frame / max(1, B1*0.7))
        title_block("one periodic 2D domain")
        ground_shadow(ax, p)
        for k in TILES:
            ix0, ix1, iy0, iy1 = tile_inset(k, 0.0)
            pc = Poly3DCollection([quad(ix0, ix1, iy0, iy1, 0)],
                                  facecolors=[rgba("#7f8a97", 0.26*p)],
                                  edgecolors="none")
            ax.add_collection3d(pc)
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0]*5, color=INK, lw=2.4, alpha=p)
        caption("a single periodic domain in x and y")
        draw_inset(False)

    # ------------------------------------------------ SPLIT: 2x2 tiles
    elif frame < B2:
        p = seg((frame - B1) / F_SPLIT, 0.0, 0.62)
        title_block("2×2 spatial decomposition")
        ground_shadow(ax, 1)
        add_plane(ax, 0, 0.9, edge_alpha=0.42, gap=0.014*p, nums=True, numsize=14)
        full_legend()
        draw_inset(True)
        caption("4 spatial subdomains  →  4 worker networks", strong=True)

    # ------------------------------------------------ EXTRUDE: stack windows
    elif frame < B3:
        title_block(r"successive time windows   ($W_1 \rightarrow W_N$)")
        ground_shadow(ax, 1)
        add_plane(ax, 0, 0.9, edge_alpha=0.42, nums=True)
        prog = (frame - B2) / (F_EXTRUDE - 8)        # last 8 frames hold
        shown = clamp01(prog) * NWIN
        for wk in range(1, NWIN+1):
            ap = ss(clamp01(shown - (wk-1)))
            if ap <= 0:
                continue
            z = wk*DZ
            add_plane(ax, z, 0.28*ap, edge_alpha=0.20*ap)
            if ap > 0.6:
                win_label(ax, wk, z, "future")
        full_legend()
        draw_inset(True)
        caption("the evolution is marched through N successive time windows")

    # ------------------------------------------------ MARCH: active window sweeps up
    elif frame < B4:
        local = frame - B3
        active_k = local // F_MARCH_W + 1
        wf = (local % F_MARCH_W) / F_MARCH_W          # 0..1 within window
        title_block("active window  —  " + wlab(active_k))
        ground_shadow(ax, 1)
        add_plane(ax, 0, 0.9, edge_alpha=0.42, nums=True)

        for wk in range(1, NWIN+1):
            if wk == active_k:
                continue
            z = wk*DZ
            done = wk < active_k
            add_plane(ax, z, 0.34 if done else 0.15, edge_alpha=0.16,
                      dark=done)
            win_label(ax, wk, z, "done" if done else "future")

        # active slab, with a small rise-in on entry
        rise = ss(wf/0.18) if wf < 0.18 else 1.0
        za = active_k*DZ
        th = 0.10 + 0.06*rise
        ztop = add_slab(ax, za, 0.96, th=th, glow=1.0)
        win_label(ax, active_k, za, "active")

        # choreography inside the window
        ga = seg(wf, 0.12, 0.42) * (1.0 - 0.35*seg(wf, 0.42, 0.72))
        for k in TILES:
            x0, x1, y0, y1 = TILES[k]
            net_glyph(ax, (x0+x1)/2, (y0+y1)/2, ztop+0.03, TILE_DARK[k],
                      scale=0.125, alpha=ga)
        ca = seg(wf, 0.40, 0.72)
        continuity_marks(ax, ztop+0.02, ca)
        seam_arcs(ax, za, ca)

        # window-to-window handoff (late)
        if active_k < NWIN:
            ha = seg(wf, 0.60, 0.9)
            if ha > 0.01:
                ax.add_artist(Arrow3D([1.08, 1.08], [0.5, 0.5],
                                      [za+0.06, za+DZ-0.06],
                                      mutation_scale=15, lw=2.6,
                                      arrowstyle="-|>", color=ACCENT, alpha=ha))
                ax.text(1.24, 0.5, za+DZ*0.5, "handoff", color=ACCENT_D,
                        fontsize=9, ha="left", va="center", alpha=ha,
                        fontweight="bold")

        full_legend()
        draw_inset(True)
        caption("one worker per tile  ·  continuity across interfaces  "
                "·  periodic seam  ·  window-to-window handoff")

    # ------------------------------------------------ FINAL: full stack
    else:
        p = ss((frame - B4) / max(1, F_FINAL*0.6))
        title_block("full space–time stack", sub_accent=False)
        ground_shadow(ax, 1)
        add_plane(ax, 0, 0.9, edge_alpha=0.42, nums=True)
        for wk in range(1, NWIN+1):
            z = wk*DZ
            add_plane(ax, z, 0.32, edge_alpha=0.18, dark=True)
            win_label(ax, wk, z, "done")
        seam_arcs(ax, 0, p)
        full_legend()
        draw_inset(True)
        caption("2×2 spatial decomposition  ·  N time windows  "
                "·  4 worker networks", strong=True)
    return []

# ----------------------------------------------------------------------------- render
if __name__ == "__main__":
    import sys
    from matplotlib.animation import FuncAnimation, PillowWriter
    out = sys.argv[1] if len(sys.argv) > 1 else "pinns_mpf_spacetime.gif"
    if out == "x":
        f = int(sys.argv[2])
        draw(f)
        fig.savefig(f"frame_{f:03d}.png", dpi=110, facecolor=BG)
        print("wrote", f"frame_{f:03d}.png", "of", NFRAMES)
    else:
        anim = FuncAnimation(fig, draw, frames=NFRAMES, interval=1000/FPS, blit=False)
        anim.save(out, writer=PillowWriter(fps=FPS), dpi=110,
                  savefig_kwargs={"facecolor": BG})
        print("wrote", out, "frames=", NFRAMES, "fps=", FPS)
