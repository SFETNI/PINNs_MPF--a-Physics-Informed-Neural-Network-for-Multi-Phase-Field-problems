"""B6 pyramidal-initialization figures - reader-facing, CPU-only, self-contained.

The pyramid is presented as one optional training mechanism:
on this single window-1 problem it reaches accuracy comparable to a flat 2x2
decomposition at comparable-to-higher cost. What it demonstrates is the *mechanism* -
interface-rich fine-box selection -> bit-exact weight transfer into a coarser parent -> explicit
seam/continuity repair -> smooth parent refinement - and it leaves the question of where such
fine-to-coarse warm-starting *helps* (deeper decompositions, larger scale separation, 3D) open.

  1. b6_pyramid_concept_w1.png            - geometry & decomposition: microstructure, phases,
                                            diffuse IC, and the three pyramid levels (junctions/worker,
                                            interface-richest selected children, 2x2 parent seam)
  2. b6_pyramid_vs_2x2_metrics_losses.png - pyramid vs flat 2x2: accuracy, health, loss terms
  3. b6_pyramid_w1_convergence.png        - supplementary: smooth parent refinement after transfer

All inputs live under this study's data/ and metrics/ directories (no external run dirs, no GPU).
"""
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
DATA = PKG / "data"
METRICS = PKG / "metrics"
OUT = PKG / "figures"
OUT.mkdir(exist_ok=True)

fd = np.load(DATA / "fd_reference_marched.npz")
frames = fd["frames"]
meta = json.loads(str(fd["meta_json"]))
_ic = np.load(DATA / "initial_condition.npz")
phi_ic = _ic["phi"]
grain_lab = _ic["lab"]           # 16 grain labels
grain_color = _ic["color"]       # grain -> softmax channel (C=6)

metrics = json.load(open(METRICS / "pyramid_w1_metrics.json", encoding="utf-8"))
pyr = metrics["pyramid_w1"]
parent = pyr["parent_finetune"]
dense = parent["dense"]
direct_metrics = json.load(open(METRICS / "direct_baseline_metrics.json", encoding="utf-8"))
direct_dense = direct_metrics["marched_direct"]["windows"][0]["dense"]


def _junction_centroids(lab):
    """Triple-junction centroids (>=3 distinct grains in a 2x2 cell). scipy if present,
    else a small pure-python 8-connected labeller so the script stays self-contained."""
    a = lab; b = np.roll(lab, -1, 0); c = np.roll(lab, -1, 1); d = np.roll(np.roll(lab, -1, 0), -1, 1)
    st = np.sort(np.stack([a, b, c, d], -1), -1)
    j = (1 + (st[..., 1] != st[..., 0]) + (st[..., 2] != st[..., 1]) + (st[..., 3] != st[..., 2])) >= 3
    try:
        from scipy import ndimage
        lb, nl = ndimage.label(j, np.ones((3, 3)))
        return [(float(ci), float(cj)) for ci, cj in ndimage.center_of_mass(j, lb, range(1, nl + 1))]
    except Exception:
        seen = np.zeros_like(j, bool); cents = []
        pts = list(zip(*np.where(j)))
        pset = set(pts)
        for p0 in pts:
            if seen[p0]:
                continue
            stack = [p0]; seen[p0] = True; blob = []
            while stack:
                r, cc = stack.pop(); blob.append((r, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        q = (r + dr, cc + dc)
                        if q in pset and not seen[q]:
                            seen[q] = True; stack.append(q)
            rs, cs = zip(*blob)
            cents.append((float(np.mean(rs)), float(np.mean(cs))))
        return cents


def _seams(ax, g, n, col, lw, ls="-"):
    for f in range(1, g):
        ax.axvline(n * f / g, color=col, lw=lw, ls=ls); ax.axhline(n * f / g, color=col, lw=lw, ls=ls)


def _clean(ax, n):
    ax.set_xlim(0, n); ax.set_ylim(0, n); ax.set_xticks([]); ax.set_yticks([])


def _mark(ax, cents, ms=6):
    if cents:
        c = np.array(cents); ax.plot(c[:, 0], c[:, 1], "o", ms=ms, mfc="w", mec="k", mew=0.9)


def _box_counts(cents, g, n):
    cnt = np.zeros((g, g), int)
    for (ci, cj) in cents:
        cnt[min(int(ci // (n / g)), g - 1), min(int(cj // (n / g)), g - 1)] += 1
    return cnt


def concept_field():
    """Geometry & decomposition of the window-1 problem, in the style of the concept gallery:
    microstructure -> phase channels -> diffuse IC, then the three pyramid levels with the
    junctions/worker load, the interface-richest selected children, and the 2x2 parent seam."""
    n = phi_ic.shape[-1]
    phase = grain_color[grain_lab]              # softmax channel per pixel (C=6)
    cents = _junction_centroids(grain_lab)
    selected = {tuple(v) for v in pyr["child_selection"]["selected"].values()}

    fig = plt.figure(figsize=(16.0, 10.4), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)

    a = fig.add_subplot(gs[0, 0])
    a.imshow(grain_lab.T, origin="lower", cmap="tab20", interpolation="nearest")
    a.plot([], []); a.set_title(f"(a) {int(_ic['n_grains'])} grains ({meta['N']}x{meta['N']}, periodic)", fontsize=12)
    _clean(a, n)

    a = fig.add_subplot(gs[0, 1])
    a.imshow(phase.T, origin="lower", cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
    a.set_title(f"(b) phase coloring -> C={int(_ic['C'])} softmax channels", fontsize=12)
    _clean(a, n)

    a = fig.add_subplot(gs[0, 2])
    im = a.imshow(phi_ic.max(0).T, origin="lower", cmap="viridis", vmin=0.33, vmax=1.0)
    _mark(a, cents, 5)
    a.set_title(r"(c) diffuse IC: $\max_c\,\phi_c$   (dark = interfaces/junctions, width $\sim\eta$)", fontsize=12)
    _clean(a, n)
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)

    levels = [(4, "4x4 = 16 workers"), (2, "2x2 = 4 workers"), (1, "1x1 = 1 worker")]
    for col, (g, name) in enumerate(levels):
        a = fig.add_subplot(gs[1, col])
        a.imshow(phase.T, origin="lower", cmap="tab10", vmin=0, vmax=9, alpha=0.42, interpolation="nearest")
        if g > 1:
            _seams(a, g, n, "k", 1.6, "--")
        _seams(a, 2, n, "r", 2.2)                 # 2x2 parent seam (transfer target) always in red
        _mark(a, cents, 6)
        cnt = _box_counts(cents, g, n)
        for i in range(g):
            for jj in range(g):
                a.text((i + 0.5) * n / g, (jj + 0.5) * n / g, str(cnt[i, jj]), ha="center", va="center",
                       fontsize=13 if g <= 2 else 10, fontweight="bold", color="k",
                       bbox=dict(boxstyle="circle,pad=0.16", fc="white", ec="k", alpha=0.85))
        if g == 4:                                 # interface-richest child that seeds each parent quadrant
            bs = n / 4
            for (bi, bj) in selected:
                a.add_patch(plt.Rectangle((bi * bs, bj * bs), bs, bs, fill=False, ec="lime", lw=3))
        jw = len(cents) / (g * g)
        a.set_title(f"({'def'[col]}) pyramid level: {name}\nJ/W mean = {jw:.1f}   (numbers = junctions/box)", fontsize=12)
        _clean(a, n)

    fig.suptitle(
        "Pyramidal initialization - how the decomposition applies (window-1 geometry, concept only)\n"
        "green = interface-richest fine box selected to seed each 2x2 parent;   red = 2x2 parent seam",
        fontsize=13.5,
    )
    fig.savefig(OUT / "b6_pyramid_concept_w1.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def convergence():
    x = np.array([r["cum_cps"] / 1e6 for r in dense])
    fd_mse = np.array([r["fd_mse"] for r in dense])
    arg = np.array([100.0 * r["fd_argmax"] for r in dense])
    diffuse = np.array([r["diffuse_band_frac"] for r in dense])
    jfrac = np.array([r["jcount_frac"] for r in dense])
    seam = np.array([r["seam_lr_max"] for r in dense])
    healthy = np.array([bool(r["healthy"]) for r in dense])
    best_i = min([i for i, r in enumerate(dense) if r["healthy"]], key=lambda i: dense[i]["fd_mse"])
    persist = pyr["window1_persistence"]["mse_to_fd"]

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
    ax[0].plot(x, fd_mse, "-o", ms=4, label="PINNs-MPF vs FD reference")
    ax[0].axhline(persist, ls="--", color="#d62728", label="persistence baseline")
    ax[0].plot(x[best_i], fd_mse[best_i], "*", ms=16, color="gold", markeredgecolor="k",
               label=f"lowest reference score seen: {dense[best_i]['fd_mse']:.2e} (not saved)")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("parent fine-tuning collocation-point-steps (millions)")
    ax[0].set_ylabel("MSE vs FD reference")
    ax[0].set_title("FD-agreement after transfer")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    ax[1].plot(x, arg, "-o", ms=4, color="#9467bd")
    ax[1].set_xlabel("parent fine-tuning collocation-point-steps (millions)")
    ax[1].set_ylabel("argmax disagreement (%)")
    ax[1].set_title("Topology disagreement")
    ax[1].grid(alpha=0.3)

    ax[2].plot(x, diffuse, "-o", ms=4, color="#2ca02c", label="diffuse-band")
    ax[2].plot(x, jfrac, "-o", ms=4, color="#ff7f0e", label="jcount fraction")
    ax[2].plot(x, seam, "-o", ms=4, color="#1f77b4", label="left/right seam max")
    ax[2].axhline(0.60, ls="--", color="#d62728", label="health limit")
    ax[2].scatter(x[healthy], diffuse[healthy], s=65, facecolors="none", edgecolors="k",
                  label="valid-field checkpoint")
    ax[2].set_xlabel("parent fine-tuning collocation-point-steps (millions)")
    ax[2].set_ylabel("fraction / max jump")
    ax[2].set_title("Health and seam repair")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

    fig.suptitle("Supplementary: parent fine-tuning is smooth and repairs the seam-broken raw transfer",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "b6_pyramid_w1_convergence.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _series(records, key, scale=1.0):
    return np.array([scale * float(r.get(key, np.nan)) for r in records])


def _best_healthy_index(records):
    healthy = [i for i, r in enumerate(records) if r.get("healthy")]
    pool = healthy if healthy else list(range(len(records)))
    return min(pool, key=lambda i: records[i]["fd_mse"])


def compare_to_direct():
    x_direct = _series(direct_dense, "cum_cps") / 1e6
    x_pyr = (pyr["l0_cps"] + _series(dense, "cum_cps")) / 1e6
    persist = pyr["window1_persistence"]["mse_to_fd"]
    fd_band_w1 = float((frames[1].max(0) < 0.95).mean())
    i_direct = _best_healthy_index(direct_dense)
    i_pyr = _best_healthy_index(dense)

    fig, ax = plt.subplots(2, 3, figsize=(16.5, 8.8))
    direct_label = "flat 2x2 decomposition"
    pyr_label = "pyramidal initialization"
    dc, pc = "#1f77b4", "#ff7f0e"

    ax[0, 0].plot(x_direct, _series(direct_dense, "fd_mse"), "-o", ms=4, color=dc, label=direct_label)
    ax[0, 0].plot(x_pyr, _series(dense, "fd_mse"), "-o", ms=4, color=pc,
                  label=f"{pyr_label} (fine-box + parent cost)")
    ax[0, 0].axhline(persist, ls="--", color="#d62728", label="persistence baseline")
    ax[0, 0].plot(x_direct[i_direct], direct_dense[i_direct]["fd_mse"], "*", ms=15, color=dc, markeredgecolor="k")
    ax[0, 0].plot(x_pyr[i_pyr], dense[i_pyr]["fd_mse"], "*", ms=15, color=pc, markeredgecolor="k")
    ax[0, 0].set_yscale("log"); ax[0, 0].set_title("FD-agreement")
    ax[0, 0].set_ylabel("MSE vs FD reference"); ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=0.3)

    ax[0, 1].plot(x_direct, _series(direct_dense, "fd_argmax", 100.0), "-o", ms=4, color=dc, label=direct_label)
    ax[0, 1].plot(x_pyr, _series(dense, "fd_argmax", 100.0), "-o", ms=4, color=pc, label=pyr_label)
    ax[0, 1].set_title("Topology disagreement"); ax[0, 1].set_ylabel("argmax disagreement (%)")
    ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=0.3)

    ax[0, 2].plot(x_direct, _series(direct_dense, "diffuse_band_frac"), "-o", ms=4, color=dc, label="flat 2x2 diffuse-band")
    ax[0, 2].plot(x_pyr, _series(dense, "diffuse_band_frac"), "-o", ms=4, color=pc, label="pyramid diffuse-band")
    ax[0, 2].plot(x_direct, _series(direct_dense, "jcount_frac"), "--", lw=1.8, color=dc, label="flat 2x2 jcount")
    ax[0, 2].plot(x_pyr, _series(dense, "jcount_frac"), "--", lw=1.8, color=pc, label="pyramid jcount")
    ax[0, 2].axhline(0.60, ls=":", color="#d62728", label="health limit")
    ax[0, 2].axhline(fd_band_w1, ls="-.", color="gray", label=f"FD band {fd_band_w1:.2f}")
    ax[0, 2].set_title("Health metrics"); ax[0, 2].set_ylabel("fraction")
    ax[0, 2].legend(fontsize=7.5); ax[0, 2].grid(alpha=0.3)

    ax[1, 0].plot(x_direct, _series(direct_dense, "total_loss"), "-", color=dc, label="flat 2x2 total")
    ax[1, 0].plot(x_pyr, _series(dense, "total_loss"), "-", color=pc, label="pyramid total")
    ax[1, 0].plot(x_direct, _series(direct_dense, "pde"), "--", color=dc, label="flat 2x2 PDE")
    ax[1, 0].plot(x_pyr, _series(dense, "pde"), "--", color=pc, label="pyramid PDE")
    ax[1, 0].set_yscale("log"); ax[1, 0].set_title("Total and PDE losses")
    ax[1, 0].set_ylabel("unweighted loss term"); ax[1, 0].legend(fontsize=7.5); ax[1, 0].grid(alpha=0.3)

    ax[1, 1].plot(x_direct, _series(direct_dense, "ic"), "-", color=dc, label="flat 2x2 IC")
    ax[1, 1].plot(x_pyr, _series(dense, "ic"), "-", color=pc, label="pyramid IC")
    ax[1, 1].plot(x_direct, _series(direct_dense, "dn"), "--", color=dc, label="flat 2x2 DN")
    ax[1, 1].plot(x_pyr, _series(dense, "dn"), "--", color=pc, label="pyramid DN")
    ax[1, 1].set_yscale("log"); ax[1, 1].set_title("IC and bulk-stabilization losses")
    ax[1, 1].legend(fontsize=7.5); ax[1, 1].grid(alpha=0.3)

    ax[1, 2].plot(x_direct, _series(direct_dense, "cont"), "-", color=dc, label="flat 2x2 continuity")
    ax[1, 2].plot(x_pyr, _series(dense, "cont"), "-", color=pc, label="pyramid continuity")
    ax[1, 2].plot(x_direct, _series(direct_dense, "pbc"), "--", color=dc, label="flat 2x2 PBC")
    ax[1, 2].plot(x_pyr, _series(dense, "pbc"), "--", color=pc, label="pyramid PBC")
    ax[1, 2].set_yscale("log"); ax[1, 2].set_title("Continuity and periodic-seam losses")
    ax[1, 2].legend(fontsize=7.5); ax[1, 2].grid(alpha=0.3)

    for a in ax.ravel():
        a.set_xlabel("collocation-point-steps (millions)")

    fig.suptitle(
        "Window 1: pyramidal initialization vs flat 2x2 - comparable accuracy at comparable-to-higher cost\n"
        "(single window, single configuration)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "b6_pyramid_vs_2x2_metrics_losses.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    concept_field()
    convergence()
    compare_to_direct()
    print("wrote pyramid figures to", OUT)
    for f in sorted(OUT.glob("b6_pyramid*.png")):
        print("  ", f.name)
