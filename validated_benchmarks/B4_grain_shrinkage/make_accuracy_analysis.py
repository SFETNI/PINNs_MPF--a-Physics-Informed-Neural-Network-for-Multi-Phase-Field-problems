"""Reader-facing B4 accuracy analysis: parabolic shrinkage law + error distribution.

Two panels, built only from the packaged interval metrics (no training, no GPU):
  (left)  R^2(t) for PINNs-MPF and the FD reference against the analytic law
          R^2 = R0^2 - 2*mu*sigma*t. Curvature-driven shrinkage is a straight line
          in R^2; the data collapsing onto it is the physical signature.
  (right) distribution of the relative radius error |R_pinn - R_ref| / R_ref over
          all accepted time steps, with median and 95th-percentile markers.

No internal run-option names appear in the figure. Interval ranges match
`make_radius_summary.py` so the two figures tell one consistent story.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIG_DIR = HERE / "figures"

# (label, file, first index (1-based), last index (1-based), absolute-time offset, color)
SEGMENTS = [
    ("early", DATA / "b4n_cap_ramp_metrics.json", 1, 79, 0.0, "#1f77b4"),
    ("middle", DATA / "b4n_geom_center_metrics.json", 67, 117, 0.0, "#2ca02c"),
    ("late", DATA / "b4n_geom_extend_metrics.json", 1, 25, 248.089, "#d62728"),
]

R0 = 25.0          # analytic initial radius (cells)
MU_SIGMA = 1.0     # mu = sigma = 1 (grid units)


def load_segment(path, i0, i1, t_offset):
    with open(path) as f:
        d = json.load(f)
    t = np.asarray(d["t_boundaries"], dtype=float)
    rp = np.asarray(d["R_pinn_cells"], dtype=float)
    rr = np.asarray(d["R_ref_cells"], dtype=float)
    s = slice(i0 - 1, i1 + 1)
    return t[s] + t_offset, rp[s], rr[s]


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.4), facecolor="white")

    # ---- Panel L: R^2(t) parabolic law ----------------------------------------
    all_rel = []
    t_max = 0.0
    for label, path, i0, i1, off, color in SEGMENTS:
        t, rp, rr = load_segment(path, i0, i1, off)
        t_max = max(t_max, float(t.max()))
        axL.plot(t, rr ** 2, color="#777777", lw=1.8, alpha=0.9, zorder=2)
        axL.plot(t, rp ** 2, color=color, lw=0, marker="o", ms=3.5,
                 label=f"PINNs-MPF ({label})", zorder=3)
        all_rel.append(np.abs(rp - rr) / rr * 100.0)

    t_line = np.linspace(0.0, t_max, 200)
    axL.plot(t_line, R0 ** 2 - 2 * MU_SIGMA * t_line, color="k", ls="--", lw=1.4,
             label=r"analytic $R^2 = R_0^2 - 2\mu\sigma t$", zorder=1)
    axL.plot([], [], color="#777777", lw=1.8, label="finite-difference reference")
    axL.set_xlabel("time")
    axL.set_ylabel(r"squared grain radius $R^2$ (cells$^2$)")
    axL.set_title("Curvature-driven shrinkage follows the parabolic law")
    axL.legend(frameon=False, fontsize=8.5, loc="upper right")
    axL.grid(True, color="#e9e9e9")
    axL.set_ylim(bottom=0)

    # ---- Panel R: relative-error distribution ---------------------------------
    rel = np.concatenate(all_rel)
    med = float(np.median(rel))
    p95 = float(np.percentile(rel, 95))
    axR.hist(rel, bins=24, color="#4c78a8", edgecolor="white", alpha=0.9)
    axR.axvline(med, color="#2ca02c", lw=1.8, label=f"median {med:.2f}%")
    axR.axvline(p95, color="#d62728", lw=1.8, ls="--", label=f"95th pct {p95:.2f}%")
    axR.set_xlabel("relative radius error |R$_{PINN}$ - R$_{FD}$| / R$_{FD}$  (%)")
    axR.set_ylabel("time-step count")
    axR.set_title(f"Error distribution over {rel.size} accepted steps")
    axR.legend(frameon=False, fontsize=9)
    axR.grid(True, axis="y", color="#e9e9e9")

    for ax in (axL, axR):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("B4 grain shrinkage - accuracy analysis (PINNs-MPF vs finite-difference reference)",
                 y=1.005, fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "b4_accuracy_analysis.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"accepted steps={rel.size}  median={med:.3f}%  95th={p95:.3f}%  max={rel.max():.3f}%")


if __name__ == "__main__":
    main()
