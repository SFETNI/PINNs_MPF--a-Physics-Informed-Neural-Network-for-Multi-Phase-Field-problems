#!/usr/bin/env python
"""
Smooth 64x64 triple-junction relaxation animation (phase-field reference).

Companion to the 128x128 relaxation GIF, built from the ACTUAL 64x64 benchmark
finite-difference reference (`reference/reference_64x64_t200.npz`) — the same
near-equilibrium microstructure shown in the 64x64 reference-vs-PINNs-MPF comparison
figure. This is a smaller-grid comparison from the same benchmark family; its initial
condition is near-equilibrium (junction angles already ~110/120/130), so the field
evolves subtly while the interfacial energy dissipates. It is NOT a 90-degree
grid-refinement control of the 128 run.

Layout matches the 128 animation: microstructure + junction-angle readout +
interfacial-energy dissipation.

Usage: python make_triple_junction_evolution_64.py   # -> media/triple_junction_relaxation_64.gif
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ  = os.path.join(HERE, "reference", "reference_64x64_t200.npz")
OUT  = os.path.join(HERE, "media", "triple_junction_relaxation_64.gif")

# ---------------------------------------------------------------- palette (matches 128)
BG, INK, SUB, FAINT = "#ffffff", "#1c232b", "#586472", "#9aa4af"
ACCENT = "#ec6f0b"
PH = ["#4472c4", "#e8820e", "#c62d2d", "#3d9e2c"]
CMAP = ListedColormap(PH)
ANG_C = ["#3b7dd8", "#8a5cc4", "#c62d2d"]
ENERGY_C = "#2f8f6b"
TAU_C = 87.890625

# ---------------------------------------------------------------- data (real 64x64 reference)
d = np.load(NPZ)
phi   = d["phi"].astype(np.float32)          # (83,4,64,64)
times = d["times"].astype(float)
ang   = d["mean_angles"].astype(float)       # (83,3), near 120
en    = d["energies"].astype(float)
en_r  = en / en[0]

# light 3-point median filter on the displayed angle curves: removes isolated
# single-frame spikes from the coarse-grid angle DETECTOR (a documented 64x64
# quantization artifact) while preserving the near-120 trend. Physics unchanged.
def _med3(a):
    out = a.copy()
    for k in range(a.shape[1]):
        out[1:-1, k] = np.median(np.stack([a[:-2, k], a[1:-1, k], a[2:, k]]), axis=0)
    return out
ang = _med3(ang)

# ---------------------------------------------------------------- smoothing (interp x2)
F = 2
labs, tt, aa, ee = [], [], [], []
for i in range(len(times) - 1):
    for s in range(F):
        a = s / F
        pi = (1 - a) * phi[i] + a * phi[i + 1]
        labs.append(pi.argmax(0).T)
        tt.append((1 - a) * times[i] + a * times[i + 1])
        aa.append((1 - a) * ang[i] + a * ang[i + 1])
        ee.append((1 - a) * en_r[i] + a * en_r[i + 1])
labs.append(phi[-1].argmax(0).T); tt.append(times[-1]); aa.append(ang[-1]); ee.append(en_r[-1])
labs = np.array(labs); tt = np.array(tt); aa = np.array(aa); ee = np.array(ee)
Nf = len(labs)
HOLD_START, HOLD_END = 8, 14
order = [0]*HOLD_START + list(range(Nf)) + [Nf-1]*HOLD_END

# ---------------------------------------------------------------- figure (matches 128)
FW, FH = 10.4, 4.9
fig = plt.figure(figsize=(FW, FH), dpi=110)
fig.patch.set_facecolor(BG)
ms_h = 0.70; ms_w = ms_h * FH / FW
axm = fig.add_axes([0.045, 0.135, ms_w, ms_h])
axa = fig.add_axes([0.545, 0.585, 0.405, 0.300])
axe = fig.add_axes([0.545, 0.170, 0.405, 0.300])

fig.text(0.045, 0.935, "Triple-junction relaxation", fontsize=15, fontweight="bold",
         color=INK)
fig.text(0.045, 0.888, "phase-field reference  ·  64 × 64  ·  near-equilibrium triple junction",
         fontsize=10.5, color=SUB)

def setup_plots():
    for ax in (axa, axe):
        ax.set_facecolor(BG)
        for sname in ("top", "right"):
            ax.spines[sname].set_visible(False)
        ax.tick_params(labelsize=7.5, colors=SUB, length=3)
        ax.set_xlim(0, 200)
    axa.set_ylim(95, 150)
    axa.axhline(120, color=FAINT, lw=1.0, ls="--")
    axa.text(2, 121.5, "120° (Young)", fontsize=7.4, color=FAINT)
    for k in range(3):
        axa.plot(times, ang[:, k], color=ANG_C[k], lw=1.6, alpha=0.9)
    axa.set_ylabel("junction angles", fontsize=8.5, color=INK)
    axa.set_title("junction angles near 120°", fontsize=8.8, color=INK, loc="left", pad=3)
    axe.set_ylim(min(en_r)*0.985, 1.01)
    axe.plot(times, en_r, color=ENERGY_C, lw=1.8)
    axe.fill_between(times, en_r, min(en_r)*0.985, color=ENERGY_C, alpha=0.10)
    axe.set_ylabel("interfacial energy  $E/E_0$", fontsize=8.5, color=INK)
    axe.set_xlabel("time  $t$", fontsize=8.5, color=INK)
    axe.set_title("interfacial energy dissipates", fontsize=8.8, color=INK, loc="left", pad=3)

def draw(fi):
    idx = order[fi]
    axm.cla(); axa.cla(); axe.cla()
    setup_plots()
    axm.imshow(labs[idx], cmap=CMAP, vmin=0, vmax=3, origin="lower",
               interpolation="nearest")
    axm.set_xticks([]); axm.set_yticks([])
    for sp in axm.spines.values():
        sp.set_edgecolor("#c4ccd6"); sp.set_linewidth(1.2)
    t = tt[idx]
    axm.set_title(f"$t = {t:5.1f}$    ({t/TAU_C:4.2f} $\\tau_c$)", fontsize=10.5,
                  color=INK, pad=5)
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
        fig.savefig(os.path.join(HERE, "media", f"_evo64_test_{sys.argv[2]}.png"),
                    dpi=110, facecolor=BG)
        print("wrote test frame")
    else:
        anim = FuncAnimation(fig, draw, frames=len(order), interval=1000/15, blit=False)
        anim.save(OUT, writer=PillowWriter(fps=15), dpi=110,
                  savefig_kwargs={"facecolor": BG})
        print("wrote", OUT, "frames=", len(order))
