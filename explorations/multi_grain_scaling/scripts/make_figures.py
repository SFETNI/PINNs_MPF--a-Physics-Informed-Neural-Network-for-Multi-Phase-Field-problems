"""B6 capability-envelope figures - reader-facing, CPU-only, inference-free, self-contained.

Builds the two framework-scaling figures from data shipped inside this package
(no external run directories, no GPU, no network inference):
  1. b6_fd_relaxation.png         - FD reference relaxation strip (90-deg brick -> 120-deg honeycomb)
  2. b6_capability_fd_vs_pinn.png - FD / PINNs-MPF / |diff| at t=0 and the marched endpoint,
                                    plus an honest per-window and whole-horizon comparison
                                    against the persistence (unevolved-field) baseline.

Honest framing (see README):
  - physics-only loss (PDE + IC + bulk stabilization + continuity + periodic-seam);
    FD is never in the loss;
  - FD-handoff marching: each of the 4 windows starts from the FD reference field at that time
    (this is not an autonomous rollout from only the t=0 field);
  - the displayed field is the checkpoint selected by a reference-based score among
    physically valid checkpoints (reference-based checkpoint selection).

All inputs live under this study's data/ and metrics/ directories.
"""
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
DATA = PKG / "data"
METRICS = PKG / "metrics"
OUT = PKG / "figures"
OUT.mkdir(exist_ok=True)

fd = np.load(DATA / "fd_reference_marched.npz")
frames, times = fd["frames"], fd["times"]              # [T,C,N,N], [T]
meta = json.loads(str(fd["meta_json"]))
phi_pinn_end = np.load(DATA / "pinns_mpf_direct_final_field.npz")["phi"]
phi_ic = np.load(DATA / "initial_condition.npz")["phi"]

cap = json.load(open(METRICS / "direct_capability_metrics.json", encoding="utf-8"))
wins = cap["marched_direct"]["windows"]
persist_h = cap["fd_meta"]["persistence_whole_horizon"]     # hold t=0 vs FD at t=100

NG, NPHASE, NJ, N = meta["K"], 6, meta["n_junctions"], meta["N"]
TAB = "tab10"


def relaxation_strip():
    fig, ax = plt.subplots(1, len(times), figsize=(3.1 * len(times), 3.4))
    for i, t in enumerate(times):
        ax[i].imshow(frames[i].argmax(0), origin="lower", cmap=TAB, vmin=0, vmax=9, interpolation="nearest")
        ax[i].set_title(f"t = {t:.0f}", fontsize=12)
        ax[i].set_xticks([]); ax[i].set_yticks([])
    fig.suptitle(
        f"FD reference - a {NG}-grain / {NPHASE}-phase 'brick wall' (90-degree junctions) "
        "relaxes toward the 120-degree honeycomb equilibrium",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "b6_fd_relaxation.png", dpi=105, bbox_inches="tight")
    plt.close(fig)


def capability_hero():
    tend = times[-1]
    fig = plt.figure(figsize=(9.6, 12.2), constrained_layout=True)
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1.0, 1.0, 1.0, 0.9])

    cols = [(phi_ic, phi_ic, f"t = 0  (initial condition)"),
            (frames[-1], phi_pinn_end, f"t = {tend:.0f}  (marched endpoint)")]
    imdiff = None
    for j, (pfd, ppinn, ttl) in enumerate(cols):
        a0 = fig.add_subplot(gs[0, j]); a1 = fig.add_subplot(gs[1, j]); a2 = fig.add_subplot(gs[2, j])
        a0.imshow(pfd.argmax(0), origin="lower", cmap=TAB, vmin=0, vmax=9, interpolation="nearest")
        a0.set_title(ttl, fontsize=12)
        a1.imshow(ppinn.argmax(0), origin="lower", cmap=TAB, vmin=0, vmax=9, interpolation="nearest")
        d = np.abs(ppinn - pfd).max(0)
        imdiff = a2.imshow(d, origin="lower", cmap="magma", vmin=0, vmax=1)
        arg = 100 * (ppinn.argmax(0) != pfd.argmax(0)).mean()
        a2.set_xlabel(f"mean |diff| = {d.mean():.3f}   argmax disagreement = {arg:.1f}%", fontsize=10)
        for a in (a0, a1, a2):
            a.set_xticks([]); a.set_yticks([])
        if j == 0:
            a0.set_ylabel("FD reference", fontsize=12)
            a1.set_ylabel("PINNs-MPF", fontsize=12)
            a2.set_ylabel("|difference|", fontsize=12)
    fig.colorbar(imdiff, ax=fig.axes[-1], fraction=0.046, pad=0.02, label="max$_c$|$\\Delta\\phi$|")

    # ---- comparison against the persistence (hold-initial-field) baseline ----
    wl = [f"W{i+1}" for i in range(len(wins))]
    x = np.arange(len(wins))
    pinn_arg = [100 * w["fd_argmax"] for w in wins]
    pers_arg = [100 * w["persistence_argmax"] for w in wins]
    pinn_mse = [w["fd_mse"] for w in wins]
    pers_mse = [w["persistence_mse"] for w in wins]

    axA = fig.add_subplot(gs[3, 0])
    axA.plot(x, pinn_arg, "o-", color="#d1495b", label="PINNs-MPF")
    axA.plot(x, pers_arg, "s--", color="#777777", label="hold handoff field")
    axA.set_xticks(x); axA.set_xticklabels(wl); axA.set_ylabel("argmax disagree (%)")
    axA.set_title("Per-window topology error vs FD", fontsize=10)
    axA.legend(frameon=False, fontsize=8); axA.grid(True, color="#ececec")

    axB = fig.add_subplot(gs[3, 1])
    axB.semilogy(x, pinn_mse, "o-", color="#d1495b", label="PINNs-MPF")
    axB.semilogy(x, pers_mse, "s--", color="#777777", label="hold handoff field")
    axB.set_xticks(x); axB.set_xticklabels(wl); axB.set_ylabel("MSE vs FD")
    axB.set_title("Per-window field error vs FD", fontsize=10)
    axB.legend(frameon=False, fontsize=8); axB.grid(True, which="both", color="#ececec")

    fig.suptitle(
        f"PINNs-MPF capability envelope - {NG} grains, {NPHASE} phases, ~{NJ} triple junctions, N={N}\n"
        "physics-only loss (FD never in the loss) - FD-handoff marching (not autonomous) - "
        "displayed field = reference-selected checkpoint",
        fontsize=10.5,
    )
    cap_line = (
        f"Whole horizon (t=0->{tend:.0f}): PINNs-MPF endpoint argmax {100*wins[-1]['fd_argmax']:.1f}% / "
        f"MSE {wins[-1]['fd_mse']:.1e}  vs  not evolving from t=0 (persistence) "
        f"argmax {100*persist_h['argmax_disagree_frac']:.1f}% / MSE {persist_h['mse_to_fd']:.1e}  "
        "-> closer to the reference than the unevolved t=0 field over the horizon.\n"
        "Per window (panels above): given the FD-handoff field, evolving it does not improve on holding it "
        f"({cap['marched_direct'].get('n_beats_persistence',0)}/{len(wins)} windows) - "
        "the marched result is re-anchored to FD each window."
    )
    fig.text(0.5, -0.015, cap_line, ha="center", va="top", fontsize=8.6, color="#333333")
    fig.savefig(OUT / "b6_capability_fd_vs_pinn.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    relaxation_strip()
    capability_hero()
    print("wrote capability figures to", OUT)
    for f in sorted(OUT.glob("b6_capability*.png")) + sorted(OUT.glob("b6_fd_*.png")):
        print("  ", f.name)
    print(f"whole-horizon: PINN {100*wins[-1]['fd_argmax']:.2f}% / {wins[-1]['fd_mse']:.2e}  "
          f"persistence {100*persist_h['argmax_disagree_frac']:.2f}% / {persist_h['mse_to_fd']:.2e}")
