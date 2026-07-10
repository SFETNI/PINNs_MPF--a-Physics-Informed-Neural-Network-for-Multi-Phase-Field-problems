"""Extended march to extinction — DEMONSTRATION media for the driven-force grain.

CPU only. No training, no reference generation. Everything is rebuilt from the packaged
PINNs-MPF phase-field snapshots (`data/phi_iv*.npy`, intervals 1..58) and metrics
(`data/metrics.json`) of the extended full-closure run. Run:

    CUDA_VISIBLE_DEVICES="" python make_extinction_media.py

This is a DEMONSTRATION of the complete shrink-to-extinction trajectory, kept deliberately
SEPARATE from the validated result in the parent folder (validated through the interval-20
target). Following the grain to R = 0 necessarily enters the sub-η regime, where the diffuse
interface can no longer resolve the grain and the field amplitude fades. The figures show that
honestly; no accuracy is claimed below R ≈ η.

Framing (same as the validated figures): the analytic sharp-interface law is drawn in a
prominent blue and the finite-difference reference (dx = 1) is kept as a distinct black dashed
curve. Both are shown; the gap is reported both ways where the radius is still resolved. As
R → 0 the *relative* gap necessarily diverges, so the near-extinction region is judged by the
ABSOLUTE gap and the field amplitude, not by a percentage.

Robust by design: the source run reached full extinction but exited on a cosmetic
divide-by-zero in its final summary line (|ΔR| / R_ref with R_ref = 0 at closure), which can
leave the metric arrays length-inconsistent. This script therefore drives everything off the
SNAPSHOTS (per-interval radius parsed from the filename, amplitude measured as max φ), which
are complete for intervals 1..58.

Outputs:
    figures/radius_to_extinction.png    radius vs time (theory prominent, FD reference, PINN),
                                        the η line, the sub-η region, dual gap-from-theory,
                                        and the max-φ amplitude panel
    figures/microstructure_closure.png  phase field at start / target / R≈η / sub-η fade / extinct
    media/full_closure.gif              animated shrinkage to extinction with live radius + amplitude
"""
import json
import os
import re
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"                 # self-contained: metrics + phase-field snapshots live here
SNAPS = DATA
FIG = HERE / "figures"; MEDIA = HERE / "media"
FIG.mkdir(parents=True, exist_ok=True); MEDIA.mkdir(parents=True, exist_ok=True)

# PINNs-MPF = green, finite-difference reference = black, analytic theory = prominent blue.
GREEN = "#2ca02c"; REF = "#000000"; THEORY_BLUE = "#1a6fd4"; ORANGE = "#e8850c"; ETA_GRAY = "#888888"

m = json.load(open(DATA / "metrics.json"))
mu, sig, dg = m["mu"], m["sigma"], abs(m["delta_g"])
eta, R0 = m["eta"], m["R0"]
cx, cy = m["center_xy"]
tb = np.array(m["t_boundaries"])
Rp_arr = np.array(m["R_pinn_cells"]); Rr_arr = np.array(m["R_ref_cells"])

# ---- build consistent per-snapshot arrays (intervals 1..N) -----------------------
def parse(fn):
    g = re.search(r"phi_iv(\d+)_R([-\d.]+)\.npy", fn.name)
    return int(g.group(1)), float(g.group(2))
snaps = sorted((parse(f)[0], f, parse(f)[1]) for f in SNAPS.glob("phi_iv*.npy"))
IV = np.array([s[0] for s in snaps])                        # 1..58
FILES = {s[0]: s[1] for s in snaps}
Rfn = np.array([s[2] for s in snaps])                       # radius encoded in filename
N = len(snaps)
T = np.array([tb[iv] for iv in IV])                         # end-time of each interval
RP = np.array([Rp_arr[iv] if iv < len(Rp_arr) else Rfn[k]   # metrics radius (idx=interval); IC at idx0
               for k, iv in enumerate(IV)])
RR = np.array([Rr_arr[iv] if iv < len(Rr_arr) else 0.0 for iv in IV])   # FD reaches 0 at closure
MP = np.array([float(np.load(FILES[iv]).max()) for iv in IV])           # measured amplitude
GAP = RR - RP
T0 = np.concatenate([[0.0], T]); RP0 = np.concatenate([[R0], RP]); RR0 = np.concatenate([[R0], RR])

TARGET = 20
sub_eta_iv = int(IV[np.argmax(RP < eta)]) if np.any(RP < eta) else int(IV[-1])
fade_iv = int(IV[np.argmax(MP < 0.9)]) if np.any(MP < 0.9) else int(IV[-1])
t_target = tb[TARGET]; t_sub = tb[sub_eta_iv]


def law_curve(tmax, h=0.004):
    ts, Rs, R = [0.0], [R0], R0
    for _ in range(int(tmax / h)):
        R += -mu * (sig / R + dg) * h
        ts.append(ts[-1] + h); Rs.append(max(R, 0.0))
        if R <= 0:
            break
    return np.array(ts), np.array(Rs)
law_t, law_R = law_curve(T[-1])
Rlaw = np.interp(T, law_t, law_R)                           # law sampled on the interval grid

# ---- dual gap-from-theory (only meaningful while the grain is resolved, R > eta) ----
resolved = RP > eta
gpl = np.abs(RP - Rlaw); gfl = np.abs(RR - Rlaw)
print(f"full closure: {N} intervals | R_pinn {R0:.1f} -> {RP[-1]:.2f} | R_ref {RR0[1]:.1f} -> {RR[-1]:.2f}")
print(f"target(20): gap-vs-FD {GAP[TARGET-1]:.3f} cell | gap-vs-theory {gpl[TARGET-1]:.3f} cell "
      f"(FD-vs-theory {gfl[TARGET-1]:.3f} cell)")
print(f"resolved region (R>eta): mean |PINN-theory| {gpl[resolved].mean():.3f} vs "
      f"mean |FD-theory| {gfl[resolved].mean():.3f} cell")
print(f"sub-eta from interval {sub_eta_iv} (R<{eta:.0f}) | amplitude fade from interval {fade_iv} (maxphi<0.9)")


def radial_profile(phi):
    xs = np.arange(phi.shape[0]); X, Y = np.meshgrid(xs, xs, indexing="ij")
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2); rb = np.arange(0, 60, 1.0)
    return rb + 0.5, np.array([phi[(r >= b) & (r < b + 1)].mean() if np.any((r >= b) & (r < b + 1))
                               else np.nan for b in rb])


def shade(ax):
    ax.axvspan(0, t_target, color=GREEN, alpha=0.06, zorder=0)
    ax.axvspan(t_sub, T[-1], color=ORANGE, alpha=0.10, zorder=0)


# ================================================================ FIG: radius to extinction
fig = plt.figure(figsize=(11, 7))
gs = GridSpec(2, 2, height_ratios=[2.3, 1], width_ratios=[1.6, 1], hspace=0.30, wspace=0.22)
axr = fig.add_subplot(gs[0, :]); shade(axr)
axr.plot(law_t, law_R, "-", color=THEORY_BLUE, lw=2.6, zorder=2,
         label="analytic sharp-interface theory  dR/dt = −μ(σ/R+|Δg|)")
axr.plot(T0, RR0, "o--", color=REF, ms=3.3, lw=1.4, zorder=3, label="finite-difference reference (dx = 1)")
axr.plot(T0, RP0, "o-", color=GREEN, ms=3.3, lw=2.0, zorder=4, label="PINNs-MPF")
axr.axhline(eta, color=ETA_GRAY, ls=":", lw=1.3)
axr.text(1.0, eta + 0.6, "η = 7 cells (interface width)", color=ETA_GRAY, fontsize=8.5)
axr.axvline(t_target, color=GREEN, ls="--", lw=1.0, alpha=.8)
axr.text(t_target - 0.5, 34, "validated\ntarget", color=GREEN, fontsize=8.5, ha="right")
axr.text(t_sub + 0.4, 12.5, "sub-η regime\n(grain under-resolved\n→ extinction)", color=ORANGE,
         fontsize=8.5, va="top")
axr.set_ylabel("grain radius (cells)"); axr.set_ylim(0, 39); axr.set_xlim(0, T[-1])
axr.legend(loc="upper right", fontsize=9); axr.grid(alpha=.25)
axr.set_title("Driven-force grain shrinkage — extended march to extinction (DEMONSTRATION)",
              fontsize=12.5, fontweight="bold")
# lower-left: dual gap from theory while resolved, absolute gap vs FD once sub-eta
axg = fig.add_subplot(gs[1, 0]); shade(axg)
axg.plot(T[resolved], gpl[resolved], "-", color=GREEN, lw=2.0, marker="o", ms=3,
         label="|PINNs-MPF − theory|")
axg.plot(T[resolved], gfl[resolved], "--", color=REF, lw=1.6, marker="s", ms=3,
         label="|finite-difference − theory|")
axg.set_ylabel("distance from\ntheory (cells)"); axg.set_xlabel("time")
axg.legend(fontsize=7.5, loc="upper left"); axg.grid(alpha=.25)
axg.set_title("while the grain is resolved (R > η), PINNs-MPF stays closest to theory", fontsize=8.5)
# lower-right: amplitude
axa = fig.add_subplot(gs[1, 1])
axa.plot(IV, MP, "o-", color=GREEN, ms=3, lw=1.6)
axa.axhline(1.0, color="0.7", ls=":", lw=1); axa.axvline(sub_eta_iv, color=ORANGE, ls="--", lw=1)
axa.text(sub_eta_iv - 1, 0.45, "R<η", color=ORANGE, ha="right", fontsize=8)
axa.set_ylim(-0.05, 1.08); axa.set_xlabel("marching interval"); axa.set_ylabel("max φ")
axa.set_title("field full-amplitude to R≈η,\nthen fades as the grain vanishes", fontsize=9); axa.grid(alpha=.25)
fig.savefig(FIG / "radius_to_extinction.png", dpi=140, bbox_inches="tight"); plt.close(fig)
print("saved figures/radius_to_extinction.png")

# ================================================================ FIG: microstructure strip
cols = [1, TARGET, sub_eta_iv, min(fade_iv + 1, int(IV[-1])), int(IV[-1])]
labels = ["start", "validated target", "R ≈ η", "sub-η fade", "extinct"]
fig, axes = plt.subplots(1, len(cols), figsize=(3.0 * len(cols), 3.3))
for ax, iv, lab in zip(axes, cols, labels):
    phi = np.load(FILES[iv]); k = int(np.where(IV == iv)[0][0])
    ax.imshow(phi.T, origin="lower", extent=[0, 128, 0, 128], cmap="viridis", vmin=0, vmax=1)
    if phi.max() > 0.5:
        ax.contour(np.arange(128), np.arange(128), phi.T, levels=[0.5], colors=GREEN, linewidths=1.6)
    ax.add_patch(Circle((cx, cy), RR[k], fill=False, ec="w", ls="--", lw=1.3))
    ax.set_title(f"{lab}\nt={T[k]:.0f}  R={RP[k]:.1f}  maxφ={MP[k]:.2f}", fontsize=8.8)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(cx - 46, cx + 46); ax.set_ylim(cy - 46, cy + 46)
fig.suptitle("Phase field from start to extinction — sharp & circular until R≈η, then the sub-η "
             "grain fades (dashed = FD reference radius)", fontsize=10.5, fontweight="bold", y=1.05)
fig.savefig(FIG / "microstructure_closure.png", dpi=140, bbox_inches="tight"); plt.close(fig)
print("saved figures/microstructure_closure.png")

# ================================================================ GIF (full closure)
try:
    from PIL import Image
    frames = []; figg = plt.figure(figsize=(8.6, 4.7))
    for k, iv in enumerate(IV):
        figg.clf(); gsg = GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.30)
        axf = figg.add_subplot(gsg[0]); axt = figg.add_subplot(gsg[1])
        phi = np.load(FILES[iv]); sub = iv >= sub_eta_iv
        axf.imshow(phi.T, origin="lower", extent=[0, 128, 0, 128], cmap="viridis", vmin=0, vmax=1)
        if phi.max() > 0.5:
            axf.contour(np.arange(128), np.arange(128), phi.T, levels=[0.5], colors=GREEN, linewidths=1.7)
        axf.add_patch(Circle((cx, cy), RR[k], fill=False, ec="w", ls="--", lw=1.5))
        axf.set_title(f"t = {T[k]:.1f}    max φ = {MP[k]:.2f}", fontsize=10.5,
                      color=(ORANGE if sub else "k"))
        axf.set_xticks([]); axf.set_yticks([]); axf.set_xlim(cx - 46, cx + 46); axf.set_ylim(cy - 46, cy + 46)
        shade(axt)
        axt.plot(law_t, law_R, "-", color=THEORY_BLUE, lw=2.2, zorder=2)
        axt.plot(T0[:k + 2], RR0[:k + 2], "o--", color=REF, ms=2.5, lw=1.2, zorder=3)
        axt.plot(T0[:k + 2], RP0[:k + 2], "o-", color=GREEN, ms=2.5, lw=1.9, zorder=4)
        axt.plot(T[k], RP[k], "o", color=(ORANGE if sub else GREEN), ms=8, zorder=5)
        axt.axhline(eta, color=ETA_GRAY, ls=":", lw=1.1)
        axt.set_xlim(0, T[-1]); axt.set_ylim(0, 39)
        axt.set_xlabel("time"); axt.set_ylabel("grain radius (cells)"); axt.grid(alpha=.25)
        axt.legend(["analytic sharp-interface theory", "finite-difference reference (dx = 1)", "PINNs-MPF"],
                   fontsize=7.5, loc="upper right")
        msg = "sub-η: interface under-resolved (known limit)" if sub else "tracking theory and the FD reference"
        figg.suptitle(f"Driven-force grain shrinkage to extinction — {msg}", fontsize=11.5, fontweight="bold")
        figg.canvas.draw()
        w, h = figg.canvas.get_width_height()
        buf = np.frombuffer(figg.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames.append(Image.fromarray(buf[:, :, :3].copy()))
    plt.close(figg)
    frames += [frames[-1]] * 8
    frames[0].save(MEDIA / "full_closure.gif", save_all=True, append_images=frames[1:],
                   duration=140, loop=0)
    print(f"saved media/full_closure.gif  ({len(frames)} frames)")
except Exception as e:
    print("GIF skipped:", repr(e))
