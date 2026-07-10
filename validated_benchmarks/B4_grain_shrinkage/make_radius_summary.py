"""Create reader-facing B4 grain-shrinkage summary figure.

This script plots only the accepted validation intervals with clear labels.
It does not use internal run-option names in the figure itself.
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

SEGMENTS = [
    ("early validated interval", DATA / "b4n_cap_ramp_metrics.json", 1, 79, 0.0, "#1f77b4"),
    ("middle validated interval", DATA / "b4n_geom_center_metrics.json", 67, 117, 0.0, "#2ca02c"),
    # The extension metrics store time relative to the FD handoff. Plot it at its
    # absolute handoff time so the figure reads as one continuous validation story.
    ("late validated interval", DATA / "b4n_geom_extend_metrics.json", 1, 25, 248.089, "#d62728"),
]


def load_segment(path, i0, i1, t_offset):
    with open(path) as f:
        d = json.load(f)
    t = np.asarray(d["t_boundaries"], dtype=float)
    rp = np.asarray(d["R_pinn_cells"], dtype=float)
    rr = np.asarray(d["R_ref_cells"], dtype=float)
    idx0 = i0 - 1
    idx1 = i1
    return t[idx0 : idx1 + 1] + t_offset, rp[idx0 : idx1 + 1], rr[idx0 : idx1 + 1]


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.5, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.08},
        facecolor="white",
    )

    for label, path, i0, i1, t_offset, color in SEGMENTS:
        t, rp, rr = load_segment(path, i0, i1, t_offset)
        rel = np.abs(rp - rr) / rr * 100.0
        axes[0].plot(t, rr, color="#777777", lw=1.8, alpha=0.9)
        axes[0].plot(t, rp, color=color, lw=2.2, label=label)
        axes[1].plot(t, rel, color=color, lw=1.7)

    axes[0].plot([], [], color="#777777", lw=1.8, label="finite-difference reference")
    axes[0].set_ylabel("grain radius (cells)")
    axes[0].legend(frameon=False, fontsize=9, ncol=2)
    axes[0].grid(True, color="#e6e6e6")

    axes[1].axhline(3.0, color="#888888", ls="--", lw=1.0, label="3% guide")
    axes[1].set_ylabel("relative error (%)")
    axes[1].set_xlabel("time")
    axes[1].grid(True, color="#e6e6e6")
    axes[1].set_ylim(bottom=0)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("B4 grain shrinkage: PINNs-MPF radius tracking against FD reference", y=0.985, fontsize=13)
    out = FIG_DIR / "b4_radius_summary.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
