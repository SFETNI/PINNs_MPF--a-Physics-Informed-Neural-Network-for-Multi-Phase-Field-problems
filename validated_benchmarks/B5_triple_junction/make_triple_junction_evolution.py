#!/usr/bin/env python
"""
Smooth triple-junction relaxation animation (phase-field reference).

Rebuilds the reader-facing B5 evolution GIF as a continuous, slow animation from the
FULL-resolution finite-difference reference trajectory (82 frames, t=0..200), rather
than a handful of hard-cut snapshots.  The microstructure relaxes from the 90-degree
initial condition toward the 120-degree Young-angle network, shown next to a live
readout of the junction-angle relaxation and interfacial-energy dissipation.

Only the phase-field reference is stored at full temporal resolution, so this animation
shows the reference physics that PINNs-MPF reproduces; the frame-by-frame PINNs-MPF vs
reference comparison is the static figure `b5_triple_junction_reference_vs_pinns_mpf_128.png`.

Data:  reference/b5_ref_90_to_120_128.npz   (phi, times, mean_angles, energies, ...)
Usage: python make_triple_junction_evolution.py   # -> media/triple_junction_relaxation_128.gif
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ  = os.path.join(HERE, "reference", "b5_ref_90_to_120_128.npz")
OUT  = os.path.join(HERE, "media", "triple_junction_relaxation_128.gif")

# ---------------------------------------------------------------- palette
BG, INK, SUB, FAINT = "#ffffff", "#1c232b", "#586472", "#9aa4af"
ACCENT = "#ec6f0b"
PH = ["#4472c4", "#e8820e", "#c62d2d", "#3d9e2c"]      # phase 1..4
CMAP = ListedColormap(PH)
ANG_C = ["#3b7dd8", "#8a5cc4", "#c62d2d"]              # 3 junction-angle curves
ENERGY_C = "#2f8f6b"
TAU_C = 87.890625

# ---------------------------------------------------------------- data + smoothing
d = np.load(NPZ)
phi   = d["phi"].astype(np.float32)          # (82,4,128,128)
times = d["times"].astype(float)             # (82,)
ang   = d["mean_angles"].astype(float)       # (82,3) -> 120
en    = d["energies"].astype(float)          # (82,)
en_r  = en / en[0]                           # E / E0

# interpolate x2 (one midpoint per interval) for silky-smooth boundary motion
F = 2
labs, tt, aa, ee = [], [], [], []
for i in range(len(times) - 1):
    for s in range(F):
        a = s / F
        pi = (1 - a) * phi[i] + a * phi[i + 1]
        labs.append(pi.argmax(0).T)          # .T + origin='lower' matches the figure
        tt.append((1 - a) * times[i] + a * times[i + 1])
        aa.append((1 - a) * ang[i] + a * ang[i + 1])
        ee.append((1 - a) * en_r[i] + a * en_r[i + 1])
labs.append(phi[-1].argmax(0).T); tt.append(times[-1]); aa.append(ang[-1]); ee.append(en_r[-1])
labs = np.array(labs); tt = np.array(tt); aa = np.array(aa); ee = np.array(ee)
N = len(labs)
HOLD_START, HOLD_END = 8, 14                 # linger on the IC and the final state
order = [0]*HOLD_START + list(range(N)) + [N-1]*HOLD_END

# ---------------------------------------------------------------- figure
FW, FH = 10.4, 4.9
fig = plt.figure(figsize=(FW, FH), dpi=110)
fig.patch.set_facecolor(BG)
# microstructure (square)
ms_h = 0.70; ms_w = ms_h * FH / FW
axm = fig.add_axes([0.045, 0.135, ms_w, ms_h])
# right-hand readouts
axa = fig.add_axes([0.545, 0.585, 0.405, 0.300])   # junction angles
axe = fig.add_axes([0.545, 0.170, 0.405, 0.300])   # energy

fig.text(0.045, 0.935, "Triple-junction relaxation", fontsize=15, fontweight="bold",
         color=INK)
fig.text(0.045, 0.888, "phase-field reference  ·  90° → 120° Young-angle network",
         fontsize=10.5, color=SUB)

def setup_plots():
    for ax in (axa, axe):
        ax.set_facecolor(BG)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=7.5, colors=SUB, length=3)
        ax.set_xlim(0, 200)
    # angles
    axa.set_ylim(80, 165)
    axa.axhline(120, color=FAINT, lw=1.0, ls="--")
    axa.text(2, 122.5, "120° (Young)", fontsize=7.4, color=FAINT)
    for k in range(3):
        axa.plot(times, ang[:, k], color=ANG_C[k], lw=1.6, alpha=0.9)
    axa.set_ylabel("junction angles", fontsize=8.5, color=INK)
    axa.set_title("angles relax toward 120°", fontsize=8.8, color=INK, loc="left",
                  pad=3)
    # energy
    axe.set_ylim(min(en_r)*0.98, 1.02)
    axe.plot(times, en_r, color=ENERGY_C, lw=1.8)
    axe.fill_between(times, en_r, min(en_r)*0.98, color=ENERGY_C, alpha=0.10)
    axe.set_ylabel("interfacial energy  $E/E_0$", fontsize=8.5, color=INK)
    axe.set_xlabel("time  $t$", fontsize=8.5, color=INK)
    axe.set_title("interfacial energy dissipates", fontsize=8.8, color=INK, loc="left",
                  pad=3)

def draw(fi):
    idx = order[fi]
    axm.cla(); axa.cla(); axe.cla()
    setup_plots()
    # microstructure
    axm.imshow(labs[idx], cmap=CMAP, vmin=0, vmax=3, origin="lower",
               interpolation="nearest")
    axm.set_xticks([]); axm.set_yticks([])
    for s in axm.spines.values():
        s.set_edgecolor("#c4ccd6"); s.set_linewidth(1.2)
    t = tt[idx]
    axm.set_title(f"$t = {t:5.1f}$    ({t/TAU_C:4.2f} $\\tau_c$)", fontsize=10.5,
                  color=INK, pad=5)
    # moving markers on the readouts
    for k in range(3):
        axa.plot([t], [aa[idx][k]], "o", color=ANG_C[k], ms=5, zorder=5,
                 markeredgecolor="white", markeredgewidth=0.8)
    axa.axvline(t, color=ACCENT, lw=1.0, alpha=0.55)
    axe.plot([t], [ee[idx]], "o", color=ENERGY_C, ms=5, zorder=5,
             markeredgecolor="white", markeredgewidth=0.8)
    axe.axvline(t, color=ACCENT, lw=1.0, alpha=0.55)
    return []

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "x":
        draw(int(sys.argv[2]))
        fig.savefig(os.path.join(HERE, "media", f"_evo_test_{sys.argv[2]}.png"),
                    dpi=110, facecolor=BG)
        print("wrote test frame")
    else:
        anim = FuncAnimation(fig, draw, frames=len(order), interval=1000/15,
                             blit=False)
        anim.save(OUT, writer=PillowWriter(fps=15), dpi=110,
                  savefig_kwargs={"facecolor": BG})
        print("wrote", OUT, "frames=", len(order))
