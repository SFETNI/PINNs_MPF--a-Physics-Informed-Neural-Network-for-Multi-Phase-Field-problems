"""Reader-facing B5 relaxation-physics figure (finite-difference reference).

Two panels, built only from the packaged reference trajectory (no training, no GPU):
  (left)  the three distinct junction angles vs time, relaxing from the 90/90/180
          initial condition toward the 120-degree Young equilibrium.
  (right) the interfacial free energy vs time, decreasing as the network relaxes.

This is the reference physics that PINNs-MPF is validated against; the per-window
PINNs-MPF-vs-reference accuracy is shown separately in the metrics figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REF = HERE / "reference" / "b5_ref_90_to_120_128.npz"
FIG_DIR = HERE / "figures"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    z = np.load(REF)
    t = np.asarray(z["times"], dtype=float)
    ang = np.asarray(z["mean_angles"], dtype=float)   # (nt, 3)
    energy = np.asarray(z["energies"], dtype=float)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.4), facecolor="white")

    # ---- Panel L: junction angles -> 120 deg ----------------------------------
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for k in range(ang.shape[1]):
        axL.plot(t, ang[:, k], color=colors[k], lw=2.0,
                 label=f"angle group {k + 1}: {ang[0, k]:.0f}$\\degree$ -> {ang[-1, k]:.0f}$\\degree$")
    axL.axhline(120.0, color="k", ls="--", lw=1.3)
    axL.text(0.5 * t[-1], 122.0, "120$\\degree$ Young equilibrium", ha="center", va="bottom",
             fontsize=9, color="#444444")
    axL.set_xlabel("time")
    axL.set_ylabel("mean junction angle (deg)")
    axL.set_title("Junction angles relax toward the 120$\\degree$ equilibrium")
    axL.legend(frameon=False, fontsize=8.5, loc="upper right")
    axL.grid(True, color="#e9e9e9")

    # ---- Panel R: free-energy dissipation -------------------------------------
    ratio = float(energy[-1] / energy[0])
    axR.plot(t, energy, color="#8e44ad", lw=2.2)
    axR.fill_between(t, energy, energy.min(), color="#8e44ad", alpha=0.08)
    axR.set_xlabel("time")
    axR.set_ylabel("interfacial free energy (arb. units)")
    axR.set_title(f"Free energy decreases to {ratio:.2f}x its initial value")
    axR.grid(True, color="#e9e9e9")
    axR.set_ylim(bottom=0)

    for ax in (axL, axR):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("B5 triple junction - relaxation physics (finite-difference reference, 128 x 128)",
                 y=1.005, fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "b5_triple_junction_relaxation_physics_128.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"wrote {out}")
    # monotonicity report for honest captioning
    diffs = np.diff(energy)
    print(f"energy start={energy[0]:.3f} end={energy[-1]:.3f} ratio={ratio:.3f} "
          f"strictly_decreasing={bool((diffs <= 1e-9).all())} n_up_steps={int((diffs > 1e-9).sum())}")
    print(f"angles start={ang[0].round(1).tolist()} end={ang[-1].round(1).tolist()}")


if __name__ == "__main__":
    main()
