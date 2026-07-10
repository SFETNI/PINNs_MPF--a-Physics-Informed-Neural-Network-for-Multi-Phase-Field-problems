"""Benchmark 2 — grain shrinkage via DOMAIN DECOMPOSITION + time-marching (correction #3).

Multi-NN: the domain is split into nx*ny overlapping sub-boxes, each with a worker network.
Boxes on the interface see a locally near-flat arc -> the local recession is easy (like B1);
continuity stitches neighbours; deep-bulk denoising pins matrix/grain; PDE only near the
interface. Moving IC time-marching across K intervals (local time tau in [0,dt]).

Produces visualizations: solution vs reference + error, radius-vs-time, and a time montage.
    python benchmarks/b2_grain_shrinkage/run_decomp.py            # full (GPU)
    python benchmarks/b2_grain_shrinkage/run_decomp.py --quick    # CPU smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pinns_mpf as pm  # noqa: E402
from pinns_mpf import io  # noqa: E402
from pinns_mpf.decomposition.decomp import (MultiNN, build_batch, decomp_fit_interval,  # noqa: E402
                                            make_boxes, make_loss)
from pinns_mpf.viz import plots  # noqa: E402
from reference.pf_solver import (curvature_law_radius, diagnose_frame, dphi_dt,  # noqa: E402
                                 grain_radius, init_circle, simulate, stable_dt)
from benchmarks.b2_grain_shrinkage.run import (ETA, L, MU, R0, SIGMA, T_FINAL,  # noqa: E402
                                               build_reference)


def build_reference_geom(p, n_ref, n_intervals, L_dom, R0_g, eta_g, t_final, center,
                         boundaries=None, fd_save_every=None):
    """Geometry-parameterized FD reference (for Benchmark-4 config: 64^2, eta=7, R0=25, etc.).
    Mirrors run.build_reference but takes explicit geometry. ``boundaries`` (optional) supplies
    NON-UNIFORM interval edges in t (adaptive dt -> fine resolution where R is small); otherwise
    uniform [0, t_final] in n_intervals steps.

    When the requested window spacing is finer than stable_dt, the FD simulation uses a refined
    timestep (min(stable_dt, min_spacing)) so reference frames align with window boundaries instead
    of snapping to the nearest coarse FD frame.  Returns fd_diag with alignment diagnostics.

    ``fd_save_every`` (optional, option-controlled): if given, the FD solver saves a frame every
    ``fd_save_every`` explicit steps.  Default (None) preserves the historical heuristic cadence
    (~4 frames per window).  Set to 1 for exact window-boundary alignment (every FD step saved),
    which drives ``fd_boundary_max_abs_error`` toward 0 at the cost of memory/time.
    """
    dx = L_dom / n_ref
    phi0, xs = init_circle(n_ref, L_dom, R0_g, eta_g, center=center)
    dt_fd_stable = stable_dt(p, dx)
    if boundaries is not None:
        bnd = np.asarray(boundaries, dtype=float)
        min_spacing = float(np.min(np.diff(bnd)))
    else:
        bnd = np.linspace(0, t_final, n_intervals + 1)
        min_spacing = float(t_final / n_intervals)
    dt_fd_used = min(dt_fd_stable, min_spacing)
    if fd_save_every is not None:
        # exact alignment: refine dt so min_spacing is an INTEGER number of FD steps. Every window
        # boundary that is a multiple of min_spacing then lands on a saved frame to machine precision.
        # Needed for non-uniform schedules where min_spacing does not divide dt_fd_stable (e.g.
        # dt=0.575 vs stable_dt=0.05 -> 11.5 steps). No-op for uniform schedules already aligned
        # (B3_DRIVEN_FORCE: 1.15/0.05=23 -> dt stays 0.05). Refinement only ever shrinks dt -> still stable.
        n_sub = int(np.ceil(min_spacing / dt_fd_used - 1e-9))
        dt_fd_used = min_spacing / n_sub
        save_every = max(1, int(fd_save_every))    # option-controlled: exact alignment at =1
    else:
        save_every = max(1, int(min_spacing / dt_fd_used / 4))   # historical heuristic default
    times, frames, _ = simulate(phi0, p, dx, dt=dt_fd_used, t_final=float(bnd[-1]), save_every=save_every)
    snap_idx = [int(np.argmin(np.abs(times - tb))) for tb in bnd]
    ref_grids = [frames[i] for i in snap_idx]
    fd_boundary_max_abs_error = float(max(abs(times[i] - tb) for i, tb in zip(snap_idx, bnd)))
    fd_diag = {"fd_dt_stable": float(dt_fd_stable), "fd_dt_used": float(dt_fd_used),
               "fd_boundary_max_abs_error": fd_boundary_max_abs_error,
               "fd_save_every": int(save_every)}
    return xs, dx, ref_grids, bnd, fd_diag


def viz_all(traj, ref_grids, xs, bnd, R_pinn, R_ref, boxes, outdir):
    os.makedirs(outdir, exist_ok=True)
    # 1) final field: pred vs reference + |error| + slices
    plots.plot_solution_vs_reference(traj[-1], ref_grids[-1], xs, xs,
                                     os.path.join(outdir, "solution_tfinal.png"))
    # 2) radius vs time (PINN vs FD reference vs sharp-interface curvature law)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(bnd, R_ref, "k-o", lw=2, ms=3, label="FD reference")
    ax.plot(bnd, R_pinn, "r--s", lw=2, ms=3, label="PINN (decomposition)")
    ax.plot(bnd, [curvature_law_radius(t, R0, MU, SIGMA) for t in bnd], "b:", lw=1.5,
            label=r"law $R^2=R_0^2-2\mu\sigma t$")
    ax.set_xlabel("t (grid time units)"); ax.set_ylabel("grain radius (cells)")
    ax.legend(); ax.set_title("B2 grain radius vs time")
    fig.savefig(os.path.join(outdir, "radius_vs_time.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    # 3) time montage: PINN (top) vs reference (bottom) at a few times
    idx = np.linspace(0, len(traj) - 1, min(5, len(traj))).astype(int)
    fig, axes = plt.subplots(2, len(idx), figsize=(3 * len(idx), 6))
    ext = [0, L, 0, L]
    for c, k in enumerate(idx):
        axes[0, c].imshow(np.clip(traj[k], 0, 1).T, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=1)
        axes[0, c].set_title(f"PINN t={bnd[k]:.0f}"); axes[0, c].set_xticks([]); axes[0, c].set_yticks([])
        axes[1, c].imshow(ref_grids[k].T, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=1)
        axes[1, c].set_title(f"ref t={bnd[k]:.0f}"); axes[1, c].set_xticks([]); axes[1, c].set_yticks([])
    # overlay box cores on the first PINN panel
    for b in boxes:
        x0, x1, y0, y1 = b["core"]
        axes[0, 0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="w", lw=0.6))
    fig.suptitle("B2 decomposition: PINN vs reference over time (boxes overlaid top-left)")
    fig.savefig(os.path.join(outdir, "montage.png"), dpi=130, bbox_inches="tight"); plt.close(fig)


def seam_diagnostics(field, xs, L_dom, nx, edges=None):
    """Max |phi| jump straddling the interior box-core seam lines, vs the global max adjacent-cell
    jump (baseline). A seam artifact shows as seam_max_jump >> baseline. ``edges`` (optional) gives
    the ACTUAL non-uniform core boundaries (e.g. [0,12,52,64]); interior edges are the seam lines.
    Without it, falls back to the uniform i*L/nx split."""
    f = np.clip(field, 0.0, 1.0)
    dxx = np.abs(np.diff(f, axis=0)); dyy = np.abs(np.diff(f, axis=1))
    baseline = float(max(dxx.max(), dyy.max())) if f.size else 0.0
    seam_pos = (list(np.asarray(edges, float)[1:-1]) if edges is not None
                else [i * L_dom / nx for i in range(1, nx)])
    seam = 0.0
    for sp in seam_pos:
        c = int(np.argmin(np.abs(np.asarray(xs) - sp)))
        c = min(max(c, 1), len(xs) - 1)
        seam = max(seam, float(dxx[c - 1].max()), float(dyy[:, c - 1].max()))
    return {"seam_max_jump": seam, "baseline_max_jump": baseline}


def extinction_summary(diag_history, eta_cells):
    """Interface-extinction time (first t with no diffuse band), numerical clearance
    time (first t with max_phi<0.5), and R^2-vs-t slope over the WELL-RESOLVED regime
    (R > 2.5*eta), for PINN and reference. Returns a dict (None where not reached)."""
    def first_t(side, pred):
        for d in diag_history:
            if pred(d[side]):
                return float(d["t"])
        return None
    out = {}
    for side in ("pinn", "ref"):
        out[f"extinction_t_{side}"] = first_t(side, lambda s: not s["interface"])
        out[f"clearance_t_{side}"] = first_t(side, lambda s: s["max_phi"] < 0.5)
        ts = np.array([d["t"] for d in diag_history
                       if d[side]["R_cells"] > 2.5 * eta_cells])
        rs = np.array([d[side]["R_cells"] for d in diag_history
                       if d[side]["R_cells"] > 2.5 * eta_cells])
        out[f"R2_slope_{side}"] = (float(np.polyfit(ts, rs ** 2, 1)[0])
                                   if len(ts) > 2 else None)
    return out


OPTIONS = {
    "A": dict(nx=4, w_ic=3.0, w_dn=6.0, w_cont=10.0, gradnorm=False,
              dn_margin_eta=0.0),  # strong balanced pinning
    "B": dict(nx=4, w_ic=2.0, w_dn=1.0, w_cont=1.0, gradnorm=True,
              dn_margin_eta=0.0),  # GradNorm adaptive balancing
    "C": dict(nx=5, w_ic=3.0, w_dn=5.0, w_cont=8.0, gradnorm=False,
              dn_margin_eta=0.0),  # finer boxes + strong pinning
    # Late distance-aware denoising to avoid small-radius over-pinning.
    "D": dict(nx=4, w_ic=3.0, w_dn=6.0, w_cont=10.0, gradnorm=False,
              dn_margin_eta=0.25, dn_margin_start_radius=34.0),
    # E: DENSE workers (paper-scale 6x128) + GradNorm adaptive loss balancing + stronger
    # denoising coverage. Fixes the full-A "grow" drift (static weights let the bulk-driving
    # PDE term win and the grain re-grows over the moving-IC intervals). GradNorm rescales each
    # term to equal gradient magnitude every gn_every Adam steps so no single term dominates;
    # the denser net resolves the thin interface profile/recession. L-BFGS history trimmed
    # (num_correction_pairs=20) so 16x(6x128) float64 fits the 8 GB A10.
    "E": dict(nx=4, w_ic=3.0, w_dn=6.0, w_cont=10.0, gradnorm=True, dn_margin_eta=0.0,
              hidden=[128, 128, 128, 128, 128, 128], n_dn=600, lbfgs_pairs=20),
    # F: PAPER-FAITHFUL — 2x2=4 networks x 6x128 (the paper's curvature benchmark uses 4 NN, 6x128),
    # STATIC strong-denoising weights (NO GradNorm: the viz showed GradNorm oscillating w_pde 26.5 -> w_dn
    # 200 between grow and freeze). Denser nets resolve the thin interface; fewer/bigger boxes make it
    # tractable (~4x faster than 16x(6x128)). Tests whether paper-scale capacity fixes the full-A grow drift.
    "F": dict(nx=2, w_ic=3.0, w_dn=6.0, w_cont=10.0, gradnorm=False, dn_margin_eta=0.0,
              hidden=[128, 128, 128, 128, 128, 128], n_dn=400, lbfgs_pairs=30),
    # G: SLOPE-CONTINUITY (user's idea) — pin d(phi)/d(tau)|tau=0 to the curvature velocity (PDE RHS
    # on the carried-forward IC) so the net continues the recession TREND across intervals instead of
    # relaxing to a static profile (the grow/freeze seen in A/E/F: only VALUE continuity was enforced,
    # never the slope). Paper-faithful 4 NN x 6x128, static strong-denoise, no GradNorm.
    "G": dict(nx=2, w_ic=3.0, w_dn=6.0, w_cont=10.0, w_slope=5.0, gradnorm=False, dn_margin_eta=0.0,
              hidden=[128, 128, 128, 128, 128, 128], n_dn=400, n_sl=200, lbfgs_pairs=30),
    # H: TREND-ENDPOINT prior (user's idea, integrated form). Pin phi(tau=dt) toward the
    # trend-extrapolated field 2*phi_k - phi_{k-1} (continue the recession the PREVIOUS interval
    # achieved). Large penalty when the grain freezes; cannot be gamed by decelerating after tau=0
    # (unlike the slope term). Interval 1 endpoint is FD-seeded (1 of 20 intervals) to bootstrap a
    # correct recession direction; intervals 2..K self-continue from the PINN's own trajectory.
    "H": dict(nx=2, w_ic=3.0, w_dn=6.0, w_cont=10.0, w_trend=10.0, gradnorm=False, dn_margin_eta=0.0,
              hidden=[128, 128, 128, 128, 128, 128], n_dn=400, n_tr=300, lbfgs_pairs=30),
    # L: LEGACY-FAITHFUL (port of legacy/Benchmark_4). Equal weights (pde=ic=cont=1), DENOISING OFF,
    # PDE on ALL points (pde_band=False), sigmoid 6x128 4NN, and -- the key fix vs from-scratch --
    # train each interval MUCH longer (toward convergence: adam 400 + L-BFGS 800) instead of the
    # under-trained 200/150. K override sets interval count (legacy uses ~150 tiny intervals; we test
    # the mechanism with more intervals than 20). No slope/trend/denoise: pure PDE+IC+continuity.
    "L": dict(nx=2, w_ic=1.0, w_dn=0.0, w_cont=1.0, gradnorm=False, dn_margin_eta=0.0,
              hidden=[128, 128, 128, 128, 128, 128], n_dn=0, lbfgs_pairs=30,
              pde_band=False, K=20, adam=400, lb=800),
    # M: HYBRID — isolates the LEGACY "train-to-convergence" fix on top of my WORKING bulk control.
    # = Option F (denoising ON to hold the bulk at moderate dt, band PDE, 4NN 6x128) but train each
    # interval MUCH longer (adam 400 + L-BFGS 800, vs F's 200/150). Cheapest clean test of whether
    # under-training (not the loss formulation) was the culprit for the grow/freeze.
    "M": dict(nx=2, w_ic=3.0, w_dn=6.0, w_cont=10.0, gradnorm=False, dn_margin_eta=0.0,
              hidden=[128, 128, 128, 128, 128, 128], n_dn=400, lbfgs_pairs=30,
              pde_band=True, K=20, adam=400, lb=800),
    # B4: PAPER BENCHMARK 4 geometry (the VALIDATED curvature-driven 4NN case): 64^2 grid, eta=7,
    # R0=25, centered, Dg=0. Modern reproduction: 4 NN (2x2) x 6x128, MANY intervals (the legacy's
    # bulk-holding mechanism) + train each interval toward convergence (adam 250 / L-BFGS 350).
    # Moderate-K verification (K=50) keeps light denoising to hold the bulk at non-tiny dt; the
    # fully-faithful K=150 no-denoise/PDE-everywhere variant follows if this recedes.
    "B4": dict(nx=2, w_ic=3.0, w_dn=6.0, w_cont=10.0, gradnorm=False, dn_margin_eta=0.0,
               hidden=[128, 128, 128, 128, 128, 128], n_dn=300, lbfgs_pairs=30,
               pde_band=True, K=80, adam=150, lb=200, n_f=3000,
               L=64.0, R0=25.0, eta=7.0, n_ref=64, t_final=240.0),
    # B4A: B4 with ADAPTIVE dt -- interval edges at uniform steps in R (t_k=(R0^2-R_k^2)/2), so dt
    # shrinks as R->0 (fine resolution in the late dive where v=mu*sigma/R is largest; coarse early).
    # Fixes the K=80 tail stall (R<16) where uniform dt=3 under-resolved the accelerating recession.
    # Per-interval dt is exact (PDE residual is in physical tau; worker tau-norm uses max dt). K=100
    # -> uniform DR~0.13 (vs ref late DR~0.19 that stalled the net). Lighter per-interval (tiny motion
    # converges fast). R_final=12 matches the K=80 horizon.
    "B4A": dict(nx=2, w_ic=3.0, w_dn=6.0, w_cont=10.0, gradnorm=False, dn_margin_eta=0.0,
                hidden=[128, 128, 128, 128, 128, 128], n_dn=300, lbfgs_pairs=30,
                pde_band=True, K=100, adam=120, lb=180, n_f=3000, adaptive_dt=True, R_final=12.0,
                L=64.0, R0=25.0, eta=7.0, n_ref=64, t_final=240.0),
    # B4T: FAST TAIL-ONLY test of the adaptive-dt fix. Isolates the hard regime -- starts at R0=16
    # (the early R25->16 already validated at <2% in B4 K=80) and runs only the R16->12 dive with
    # adaptive fine dt. ~30 intervals (~1.5h) vs the full ~6h. Answers "does adaptive-dt resolve the
    # tail?" without re-solving the easy part. Same nets/loss/training as B4A.
    "B4T": dict(nx=2, w_ic=3.0, w_dn=6.0, w_cont=10.0, gradnorm=False, dn_margin_eta=0.0,
                hidden=[128, 128, 128, 128, 128, 128], n_dn=300, lbfgs_pairs=30,
                pde_band=True, K=30, adam=120, lb=180, n_f=3000, adaptive_dt=True, R_final=12.0,
                L=64.0, R0=16.0, eta=7.0, n_ref=64, t_final=56.0),
    # ── Adaptive-geometry handoff ───────────────────────────────────────────────────────────────
    # Large-radius interval: B4N_CAP (2×2, validated R=25→17.5, <0.6% error) runs as normal.
    # Tail interval: center-large 3×3 [0,12,52,64] continues from the CAP PINN field at R≈16,
    # loaded as IC via phi_init_path (no FD labels — the carried PINN field is the IC).
    # Grain is 100% inside [12,52] for R≤16 → no seam-cross, clean deep-tail tracking.
    # B4N_GEOM_TAIL_SMOKE: 8-interval smoke to check the IC-transfer transient.
    # B4N_GEOM_TAIL: full tail run R≈16→11 (~42 intervals).
    # B4N_GEOM_TAIL_SMOKE: 8-interval PHYSICS smoke at validated capacity (ΔR≈0.117/interval).
    # R_final=15.1 so total recession ≈ 0.9 cells — stays within the ~0.117 cells/interval capacity
    # proven by B4N_CAP. Tests the IC-transfer transient ONLY; not the full tail.
    # Acceptance target: relerr ≤ 1.5%, maxphi ≥ 0.995, mass not growing after iv3.
    # B4N_GEOM_TAIL_SMOKE: 8-interval physics smoke at proper training budget.
    # R0=15.8 matches handoff field R_meas=15.73 so reference FD starts near the handoff.
    # K=8, R_final=15.1 → ΔR=0.0875/interval (below capacity ~0.117 → should converge cleanly).
    # adam=120, lb=180: same budget as full runs (needed: 9 workers × 100K params can't converge in 40+40).
    # Acceptance target: relerr ≤ 1.5%, maxφ ≥ 0.995, mass stable after iv3.
    # B4N_GEOM_TAIL_SMOKE: warm-started from B4N_GEOM_CENTER final weights (same 3x3 architecture).
    # The GC weights (last interval at R≈11.2) give the optimizer a strong starting basin for the
    # center-large geometry. The 120 Adam steps adjust to the handoff IC (R=15.8); L-BFGS polishes.
    # Acceptance target: relerr ≤ 1.5%, maxφ ≥ 0.995, mass stable after iv3.
    "B4N_GEOM_TAIL_SMOKE": dict(
        nx=3, hidden=[128, 128, 128, 128, 128, 128], lbfgs_pairs=30,
        pde_band=True, pde_bulk_frac=0.0, project=False, periodic_embed=False,
        box_edges=[0.0, 12.0, 52.0, 64.0],
        mu=1.0, sigma=1.0, delta_g=0.0, eta=7.0, R0=15.8, L=64.0, n_ref=64, units="grid",
        w_ic=3.0, w_cont=10.0, w_dn=6.0, n_dn=300, dn_margin_eta=1.0,
        dn_margin_start_radius=16.5, dn_margin_ramp_width=4.0,
        gradnorm=False, K=8, adam=120, lb=180, n_f=3000,
        adaptive_dt=True, R_final=15.1, t_final=312.5,
        phi_init_path="outputs/b4n_geom_handoff/handoff_phi.npy",
        warm_weights_path="outputs/b4n_geom_center/weights_warm.npy"),
    # B4N_GEOM_TAIL: full tail R≈15.8→11 (K=55, ΔR=0.087/interval — below capacity → safe).
    # Full budget (adam=120, lb=180) + warm start from GC weights — reference quality run.
    "B4N_GEOM_TAIL": dict(
        nx=3, hidden=[128, 128, 128, 128, 128, 128], lbfgs_pairs=30,
        pde_band=True, pde_bulk_frac=0.0, project=False, periodic_embed=False,
        box_edges=[0.0, 12.0, 52.0, 64.0],
        mu=1.0, sigma=1.0, delta_g=0.0, eta=7.0, R0=15.8, L=64.0, n_ref=64, units="grid",
        w_ic=3.0, w_cont=10.0, w_dn=6.0, n_dn=300, dn_margin_eta=1.0,
        dn_margin_start_radius=16.5, dn_margin_ramp_width=4.0,
        gradnorm=False, K=55, adam=120, lb=180, n_f=3000,
        adaptive_dt=True, R_final=11.0, t_final=312.5,
        phi_init_path="outputs/b4n_geom_handoff/handoff_phi.npy",
        warm_weights_path="outputs/b4n_geom_center/weights_warm.npy"),
    # B4N_GEOM_TAIL_FAST: speed-optimised tail for single GPU window.
    # Warm-started from B4N_GEOM_CENTER weights (same architecture). lb=60, lbfgs_pairs=20.
    # Estimated: ~4-6 min/interval × K=55 = 3.5-5.5h → fits comfortably in overnight window.
    "B4N_GEOM_TAIL_FAST": dict(
        nx=3, hidden=[128, 128, 128, 128, 128, 128], lbfgs_pairs=20,
        pde_band=True, pde_bulk_frac=0.0, project=False, periodic_embed=False,
        box_edges=[0.0, 12.0, 52.0, 64.0],
        mu=1.0, sigma=1.0, delta_g=0.0, eta=7.0, R0=15.8, L=64.0, n_ref=64, units="grid",
        w_ic=3.0, w_cont=10.0, w_dn=6.0, n_dn=300, dn_margin_eta=1.0,
        dn_margin_start_radius=16.5, dn_margin_ramp_width=4.0,
        gradnorm=False, K=55, adam=120, lb=60, n_f=3000,
        adaptive_dt=True, R_final=11.0, t_final=312.5,
        phi_init_path="outputs/b4n_geom_handoff/handoff_phi.npy",
        warm_weights_path="outputs/b4n_geom_center/weights_warm.npy"),
    # B4N_GEOM_SNAP: snapshot collection run for 2D geometry figures.
    # K=12 adaptive-dt intervals (R=25→12, uniform in R), adam=100, lb=60. Warm-started from
    # GEOM_CENTER weights_warm (R≈13). Saves phi array every 2 intervals → 6 snapshots at R≈23,
    # 21, 19, 17, 15, 13. w_ic=8 strong to keep IC propagation stable despite warm-start mismatch.
    "B4N_GEOM_SNAP": dict(
        nx=3, hidden=[128, 128, 128, 128, 128, 128], lbfgs_pairs=20,
        pde_band=True, pde_bulk_frac=0.0, project=False, periodic_embed=False,
        box_edges=[0.0, 12.0, 52.0, 64.0],
        mu=1.0, sigma=1.0, delta_g=0.0, eta=7.0, R0=25.0, L=64.0, n_ref=64, units="grid",
        w_ic=8.0, w_cont=10.0, w_dn=6.0, n_dn=300, dn_margin_eta=1.0,
        dn_margin_start_radius=20.0, dn_margin_ramp_width=4.0,
        gradnorm=False, K=12, adam=100, lb=60, n_f=3000,
        adaptive_dt=True, R_final=12.0, t_final=312.5,
        warm_weights_path="outputs/b4n_geom_center/weights_warm.npy",
        save_phi_every=2),
    # B4N_GEOM_CENTER (center-large geometry): the one untested DEEP-TAIL
    # lever. Non-uniform 3x3 edges [0,12,52,64] -> central core [12,52] (half-width 20 cells) holds the
    # WHOLE shrinking grain for R<=20, so it is NOT split across the seam cross through the failure band
    # (equal-3x3 central half-width is only 10.67 -> still split; rejected). Otherwise the validated
    # B4N_CAP_RAMP recipe: capacity-matched adaptive stepping, gradual denoise release, project=False,
    # pde_bulk_frac=0, NO w_slope, NO w_profile (pure geometry diagnostic). Unsupervised after IC.
    "B4N_GEOM_CENTER": dict(
        nx=3, hidden=[128, 128, 128, 128, 128, 128], lbfgs_pairs=30,
        pde_band=True, pde_bulk_frac=0.0, project=False, periodic_embed=False,
        box_edges=[0.0, 12.0, 52.0, 64.0],
        mu=1.0, sigma=1.0, delta_g=0.0, eta=7.0, R0=25.0, L=64.0, n_ref=64, units="grid",
        w_ic=3.0, w_cont=10.0, w_dn=6.0, n_dn=300, dn_margin_eta=1.0, dn_margin_start_radius=16.0,
        dn_margin_ramp_width=4.0, gradnorm=False, K=120, adam=120, lb=180, n_f=3000,
        adaptive_dt=True, R_final=11.0, t_final=312.5),
    # B4N_GEOM_EXTEND: extension of GEOM_CENTER from the FD field at t=248 (R≈11.95) down to R≈8.5.
    # Starts from a single FD IC handoff (phi_iv117_fd.npy); all subsequent ICs are PINN-propagated.
    # Same 3x3 center-large geometry — grain fully inside central core for R<=20.
    # Warm-start from GEOM_CENTER weights_warm (iv81 state) — spatial structure is reused;
    # temporal normalization mismatch (dt_max_gc=2.91 vs dt_max_ext≈1.61) is corrected by Adam+LBFGS.
    # w_ic=5.0 (stronger than 3.0) to guard against rapid-shrinkage local minimum at small R.
    # dn_margin fully active from interval 1 (dn_margin_start_radius=25.0 > any R in this run).
    "B4N_GEOM_EXTEND": dict(
        nx=3, hidden=[128, 128, 128, 128, 128, 128], lbfgs_pairs=30,
        pde_band=True, pde_bulk_frac=0.0, project=False, periodic_embed=False,
        box_edges=[0.0, 12.0, 52.0, 64.0],
        mu=1.0, sigma=1.0, delta_g=0.0, eta=7.0, R0=11.95, L=64.0, n_ref=64, units="grid",
        w_ic=5.0, w_cont=10.0, w_dn=6.0, n_dn=300, dn_margin_eta=1.0,
        dn_margin_start_radius=25.0, dn_margin_ramp_width=4.0,
        gradnorm=False, K=25, adam=120, lb=180, n_f=3000,
        adaptive_dt=True, R_final=8.5, t_final=40.0,
        phi_init_path=os.path.join(ROOT, "validated_benchmarks", "B4_grain_shrinkage", "data", "phi_iv117_fd.npy"),
        warm_weights_path=os.path.join(ROOT, "validated_benchmarks", "B4_grain_shrinkage", "data", "b4n_geom_center_weights_warm.npy"),
        save_phi_every=2),
    # B4N_GEOM_SMOKE: 2-interval DRY CHECK of the center-large geometry (structure/routing/no-crash,
    # NOT physics). Verifies the grain routes to the central worker before any full run. Tiny budget.
    "B4N_GEOM_SMOKE": dict(
        nx=3, hidden=[128, 128, 128, 128, 128, 128], lbfgs_pairs=20,
        pde_band=True, pde_bulk_frac=0.0, project=False, periodic_embed=False,
        box_edges=[0.0, 12.0, 52.0, 64.0],
        mu=1.0, sigma=1.0, delta_g=0.0, eta=7.0, R0=25.0, L=64.0, n_ref=64, units="grid",
        w_ic=3.0, w_cont=10.0, w_dn=6.0, n_dn=200, dn_margin_eta=1.0, dn_margin_start_radius=16.0,
        dn_margin_ramp_width=4.0, gradnorm=False, K=2, adam=40, lb=40, n_f=2000,
        adaptive_dt=True, R_final=11.0, t_final=312.5),
    # ---------------------------------------------------------------------------------------------
    # B4N_* : REDESIGNED curvature-driven (Delta_g=0) benchmark -> run to EXTINCTION, unsupervised.
    # Key formulation change vs B4/B4A: PROJECTED EFFECTIVE RHS (project=True) makes the saturated
    # bulk physically inert (matches the clipped FD reference), so denoising becomes a light,
    # DISTANCE-GATED safety control (dn_margin_eta=1.0 -> pins only deep-bulk cells >=1*eta from the
    # interface, so the shrinking core is RELEASED late -- fixes the over-pinning stall hypothesis H2).
    # Domain-periodic worker embedding (periodic_embed) aligns the model BCs with the periodic
    # reference. Adaptive dt (uniform steps in R) resolves the accelerating late dive. No PF labels
    # after the IC. See docs/B4N_REDESIGN_REPORT.md.
    #
    # B4N_B4GEOM_EXT: original B4 grid-unit geometry (eta=7, R0=25, mu=sigma=1) continued to
    # extinction (sharp t_ext=R0^2/2=312.5). Directly comparable to the prior B4 results.
    # CORRECTED baseline (v2): the v1 config (w_dn=2, bulk_frac=0.5, periodic on) REGRESSED the early
    # tracking (gutted core-holding -> maxphi sagged at R=24, radius oscillated, ~3-5% by interval 8).
    # v2 = validated B4A recipe (strong core-holding denoise) + ONE targeted anti-stall change:
    # distance-margin denoising activates only BELOW R=16 (dn_margin_start_radius), so early/mid stays
    # tight (like old B4: ~0.2%, maxphi=1) and only the small-radius core is released. Pure band PDE
    # (bulk_frac=0) restores front resolution; projected RHS kept as harmless safety; periodic off.
    "B4N_B4GEOM_EXT": dict(
        nx=2, hidden=[128, 128, 128, 128, 128, 128], lbfgs_pairs=30,
        pde_band=True, pde_bulk_frac=0.0,
        mu=1.0, sigma=1.0, delta_g=0.0, eta=7.0, R0=25.0, L=64.0, n_ref=64, units="grid",
        project=True, pde_eps=1e-3, periodic_embed=False,
        w_ic=3.0, w_cont=10.0, w_dn=6.0, n_dn=300, dn_margin_eta=1.0, dn_margin_start_radius=16.0,
        gradnorm=False, K=110, adam=120, lb=180, n_f=3000,
        adaptive_dt=True, R_final=3.5, t_final=312.5),
    # B4N_CAP: CAPACITY-MATCHED radius run = validated B4A recipe (project=False, strong core-holding
    # denoise) + the ONE change that distinguished 0.2%-tracking B4A from the under-shrinking trials:
    # smaller per-interval recession (ΔR≈0.117 cells, within the PINN's ~0.13 capacity) via more
    # intervals (K=120 to R=11), PLUS late-only core release (dn_margin_start_radius=16). Tests whether
    # capacity-matched stepping carries the radius past the original R<16 stall toward R≈11 at <10%.
    # Robust (no finicky per-interval energy criterion); free energy is logged per interval as a MONITOR.
    "B4N_CAP": dict(
        nx=2, hidden=[128, 128, 128, 128, 128, 128], lbfgs_pairs=30,
        pde_band=True, pde_bulk_frac=0.0,
        mu=1.0, sigma=1.0, delta_g=0.0, eta=7.0, R0=25.0, L=64.0, n_ref=64, units="grid",
        project=False, periodic_embed=False,
        w_ic=3.0, w_cont=10.0, w_dn=6.0, n_dn=300, dn_margin_eta=1.0, dn_margin_start_radius=16.0,
        gradnorm=False, K=120, adam=120, lb=180, n_f=3000,
        adaptive_dt=True, R_final=11.0, t_final=312.5),
    # B4N_CAP_BULK (architecture-review Plan A): tests whether the late under-shrink below R~16 is
    # caused by the HARD denoise release (which removed phi=1 core control with NO replacement, since
    # B4N_CAP had project=False & pde_bulk_frac=0) rather than the intrinsic sub-2.5eta limit. Same as
    # B4N_CAP but adds a REPLACEMENT bulk-control mechanism: projected RHS (project=True) + light bulk
    # collocation (pde_bulk_frac=0.15), and replaces the hard denoise-margin switch at R=16 with a
    # GRADUAL ramp of the margin 0->1*eta over R=16->12 (dn_margin_ramp_width=4.0). Unsupervised.
    "B4N_CAP_BULK": dict(
        nx=2, hidden=[128, 128, 128, 128, 128, 128], lbfgs_pairs=30,
        pde_band=True, pde_bulk_frac=0.15,
        mu=1.0, sigma=1.0, delta_g=0.0, eta=7.0, R0=25.0, L=64.0, n_ref=64, units="grid",
        project=True, pde_eps=1e-3, periodic_embed=False,
        w_ic=3.0, w_cont=10.0, w_dn=6.0, n_dn=300, dn_margin_eta=1.0, dn_margin_start_radius=16.0,
        dn_margin_ramp_width=4.0, gradnorm=False, K=120, adam=120, lb=180, n_f=3000,
        adaptive_dt=True, R_final=11.0, t_final=312.5),
    # B4N_CAP_RAMP: clean isolation of the reviewer's core claim (is the HARD denoise release at R=16
    # the culprit for the sub-16 degradation?). Identical to B4N_CAP (project=False, pde_bulk_frac=0,
    # w_dn=6) EXCEPT the denoise margin is released GRADUALLY (ramp 0->1*eta over R=16->12) instead of
    # the hard 0->eta switch at R=16. No project/bulk over-shrink confound. If the sub-16 tail tracks
    # better than B4N_CAP (which jumped 1.8%->5.2% right after the R=16 release), the hard release
    # contributed; if it still breaks at ~2.5eta, the limit is intrinsic.
    "B4N_CAP_RAMP": dict(
        nx=2, hidden=[128, 128, 128, 128, 128, 128], lbfgs_pairs=30,
        pde_band=True, pde_bulk_frac=0.0,
        mu=1.0, sigma=1.0, delta_g=0.0, eta=7.0, R0=25.0, L=64.0, n_ref=64, units="grid",
        project=False, periodic_embed=False,
        w_ic=3.0, w_cont=10.0, w_dn=6.0, n_dn=300, dn_margin_eta=1.0, dn_margin_start_radius=16.0,
        dn_margin_ramp_width=4.0, gradnorm=False, K=120, adam=120, lb=180, n_f=3000,
        adaptive_dt=True, R_final=11.0, t_final=312.5),
    # B4N_PHASE64: PINN-Phase natural normalized convention. Domain [0,1]^2, eta=7/64, R0=0.30,
    # mu=5e-5, sigma=1. Same curvature physics in normalized units; sharp t_ext=R0^2/(2*mu)=900.
    "B4N_PHASE64": dict(
        nx=2, hidden=[128, 128, 128, 128, 128, 128], lbfgs_pairs=30,
        pde_band=True, pde_bulk_frac=0.5,
        mu=5.0e-5, sigma=1.0, delta_g=0.0, eta=7.0 / 64.0, R0=0.30, L=1.0, n_ref=64,
        units="normalized",
        project=True, pde_eps=1e-3, periodic_embed=True,
        w_ic=3.0, w_cont=10.0, w_dn=2.0, n_dn=250, dn_margin_eta=1.0, gradnorm=False,
        K=110, adam=120, lb=180, n_f=4000,
        adaptive_dt=True, R_final=0.0547, t_final=900.0),
    # B4N_SMOKE: tiny smoke of the redesigned pipeline (units/sign/BC/bounded fields).
    "B4N_SMOKE": dict(
        nx=2, hidden=[48, 48, 48], lbfgs_pairs=20,
        pde_band=True, pde_bulk_frac=0.5,
        mu=1.0, sigma=1.0, delta_g=0.0, eta=7.0, R0=25.0, L=64.0, n_ref=64, units="grid",
        project=True, pde_eps=1e-3, periodic_embed=True,
        w_ic=3.0, w_cont=10.0, w_dn=2.0, n_dn=200, dn_margin_eta=1.0, gradnorm=False,
        K=4, adam=40, lb=40, n_f=1500, adaptive_dt=True, R_final=20.0, t_final=100.0),
    # ─────────────────────────────────────────────────────────────────────────
    # B3 — DRIVING-FORCE GRAIN SHRINKAGE (paper Benchmark 3)
    # ─────────────────────────────────────────────────────────────────────────
    # The VALIDATED benchmark option is "B3_DRIVEN_FORCE" (further below): corrected-
    # scale physics delta_g=-0.5 (Pe=19), R0=38, 128x128, 3x3 non-uniform workers, with
    # the per-window Adam<->L-BFGS convergence gate. Use it to reproduce the B3 evidence.
    #
    # "B3" and "B3_SMOKE" here are EXPLORATORY ONLY (strong-driving delta_g=-250, 2x2
    # workers, single Adam->L-BFGS pass, no convergence gate) — a documented dead-end kept
    # for provenance. They are NOT the validated result and NOT the default. The validated
    # evidence lives in validated_benchmarks/B3_driven_force/.
    # B3 (exploratory): delta_g=-250 gives a near-linear reference; t_final=0.120 stops
    # just before the grain reaches R≈2*eta.
    "B3": dict(
        nx=2, gradnorm=False,
        hidden=[64, 64, 64, 64, 64, 64], lbfgs_pairs=50,
        mu=1.0, sigma=1.0, delta_g=-250.0, eta=7.0, R0=25.6, L=64.0, n_ref=64, units="grid",
        project=True, pde_eps=1e-3,
        pde_band=True, pde_bulk_frac=0.0,
        w_ic=3.0, w_dn=6.0, w_cont=10.0,
        n_dn=400, dn_margin_eta=0.0,
        K=10, adam=800, lb=0, n_f=2400,
        t_final=0.120),
    # B3_SMOKE: first-window sanity check only.
    "B3_SMOKE": dict(
        nx=2, gradnorm=False,
        hidden=[48, 48, 48], lbfgs_pairs=20,
        mu=1.0, sigma=1.0, delta_g=-250.0, eta=7.0, R0=25.6, L=64.0, n_ref=64, units="grid",
        project=True, pde_eps=1e-3,
        pde_band=True, pde_bulk_frac=0.0,
        w_ic=3.0, w_dn=6.0, w_cont=10.0,
        n_dn=200, dn_margin_eta=0.0,
        K=1, adam=200, lb=0, n_f=1500,
        t_final=0.012),
    # ─────────────────────────────────────────────────────────────────────────
    # B3_DRIVEN_FORCE — the VALIDATED driving-force benchmark (paper Benchmark 3)
    # ─────────────────────────────────────────────────────────────────────────
    # Corrected-scale physics (delta_g=-0.5 => Pe=|delta_g|*R0/sigma=19, moderate driving)
    # on a 128x128 grid, grain R0=38 cells, 3x3 NON-UNIFORM workers with core edges
    # [0,20,108,128] (one large central worker holds the grain, thin outer workers hold the
    # matrix) and overlap=7. The interface is held by the per-window legacy convergence
    # protocol: repeat [Adam(3000) -> L-BFGS(500)] up to cycles=8 times per window, advancing
    # only once the total weighted loss falls below loss_thresh=2e-6, and keeping the best-loss
    # cycle's weights (restore_best_cycle=True). Marched K=40 uniform windows (dt=1.15) to t=46
    # with exact FD window-boundary alignment (fd_save_every=1). Physics/geometry/loss/training
    # keys are identical to the config that produced validated_benchmarks/B3_driven_force/.
    "B3_DRIVEN_FORCE": dict(
        nx=3, gradnorm=False,
        hidden=[64, 64, 64, 64, 64, 64], lbfgs_pairs=50,
        mu=1.0, sigma=1.0, delta_g=-0.5, eta=7.0, R0=38.0, L=128.0, n_ref=128, units="grid",
        box_edges=[0.0, 20.0, 108.0, 128.0], overlap=7.0,
        project=True, pde_eps=1e-3,
        pde_band=True, pde_bulk_frac=0.0,
        w_ic=1.0, w_dn=6.0, w_cont=5.0,
        n_dn=400, dn_margin_eta=0.0,
        K=40, adam=3000, lb=500, n_f=2400,
        cycles=8, adam_per_cycle=None, loss_thresh=2e-6,
        restore_best_cycle=True,
        save_phi_every=1,
        t_final=46.0, fd_save_every=1,
        geometry_label="B3 driving-force grain shrinkage (validated): corrected-scale physics "
                       "delta_g=-0.5 (Pe=19), R0=38, 128x128, 3x3 non-uniform workers "
                       "(edges [0,20,108,128], overlap=7), per-window Adam<->L-BFGS convergence "
                       "gating to loss<2e-6 with best-cycle restore; K=40 windows (dt=1.15) to t=46"),
}


def run(quick=False, outdir=None, seed=1234, option="A", dn_margin_eta=None, resume=False,
        handoff_at=None):
    global L, R0, ETA, T_FINAL, MU, SIGMA  # per-option geometry + units override
    o = OPTIONS[option]; nx = ny = o["nx"]; gradnorm = o["gradnorm"]
    dn_margin_eta = o.get("dn_margin_eta", 0.0) if dn_margin_eta is None else dn_margin_eta
    dn_margin_start_radius = o.get("dn_margin_start_radius")
    dn_margin_ramp_width = o.get("dn_margin_ramp_width")  # if set, ramp margin 0->base over this dR
    if quick:
        n_ref, K, n_f, n_ic, n_dn, n_cont, adam, lb = 64, 6, 1500, 150, 150, 80, 100, 80
        hidden = [48, 48, 48]
    else:
        n_ref, K, n_f, n_ic, n_dn, n_cont, adam, lb = 128, 20, 6000, 300, 300, 150, 200, 150
        hidden = [48, 48, 48, 48]
    # per-option overrides (architecture / denoising coverage / L-BFGS history)
    hidden = o.get("hidden", hidden)
    n_dn = o.get("n_dn", n_dn)
    lbfgs_pairs = o.get("lbfgs_pairs", 50)
    w_slope = o.get("w_slope", 0.0)
    n_sl = o.get("n_sl", 0)
    w_trend = o.get("w_trend", 0.0)
    n_tr = o.get("n_tr", 0)
    pde_band = o.get("pde_band", True)
    K = o.get("K", K)
    adam = o.get("adam", adam)
    lb = o.get("lb", lb)
    n_f = o.get("n_f", n_f)
    # per-option GEOMETRY + UNITS override (B4 paper config / B4N normalized): module globals so
    # run() + viz_all both see it
    L = o.get("L", L); R0 = o.get("R0", R0); ETA = o.get("eta", ETA); T_FINAL = o.get("t_final", T_FINAL)
    MU = o.get("mu", MU); SIGMA = o.get("sigma", SIGMA)
    n_ref = o.get("n_ref", n_ref)
    delta_g = o.get("delta_g", 0.0)
    project = o.get("project", False)            # bound-aware effective RHS (matches clipped reference)
    pde_eps = o.get("pde_eps", 1e-3)
    pde_bulk_frac = o.get("pde_bulk_frac", 0.0)  # fraction of PDE collocation in the bulk (for projection)
    periodic_embed = o.get("periodic_embed", False)  # full-domain sin/cos worker embedding
    units_label = o.get("units", "grid")
    w_profile = o.get("w_profile", 0.0)              # interface-profile regularizer weight (0 = off)
    box_edges = o.get("box_edges")                   # non-uniform core edges, e.g. [0,12,52,64] (center-large)
    cycles = o.get("cycles", 1)                      # opt-in Adam<->L-BFGS convergence cycling (default 1 = legacy single pass)
    adam_per_cycle = o.get("adam_per_cycle", None)   # Adam steps per cycle (None = adam)
    loss_thresh = o.get("loss_thresh", None)         # early-break when total weighted loss < thresh
    restore_best_cycle = o.get("restore_best_cycle", False)  # opt-in: keep argmin-loss cycle weights

    info = pm.setup(seed=seed)
    print("env:", info)
    p = pm.PhysicalParams(mu=MU, sigma=SIGMA, eta=ETA, delta_g=delta_g)
    dx_print = L / n_ref
    print(f"UNIT CONVENTION [{units_label}]: domain [0,{L}]^2  grid {n_ref}x{n_ref}  dx={dx_print:.6g}  "
          f"eta={ETA:.6g} ({ETA/dx_print:.3g} cells)  R0={R0:.6g} ({R0/dx_print:.3g} cells)  "
          f"mu={MU:.6g}  sigma={SIGMA:.6g}  delta_g={delta_g:.6g}  "
          f"sharp t_ext=R0^2/(2*mu*sigma)={R0**2/(2.0*MU*SIGMA):.6g}  horizon t_final={T_FINAL:.6g}")
    print(f"FORMULATION: projected_RHS={project} (eps={pde_eps})  pde_band={o.get('pde_band', True)} "
          f"pde_bulk_frac={pde_bulk_frac}  periodic_embed={periodic_embed}  "
          f"denoise w_dn={o['w_dn']} dn_margin_eta={dn_margin_eta} start_R={dn_margin_start_radius} "
          f"ramp_width={dn_margin_ramp_width} (active >= {float(dn_margin_eta)*ETA:.3g} from interface)  "
          f"UNSUPERVISED after IC (no PF labels)")
    adaptive = o.get("adaptive_dt", False)
    if adaptive:  # interval edges at uniform steps in R -> dt shrinks as R->0 (fine late dive)
        Rk = np.linspace(R0, o.get("R_final", 12.0), K + 1)
        boundaries = (R0 ** 2 - Rk ** 2) / (2.0 * MU * SIGMA)  # t_k from R^2 = R0^2 - 2*mu*sigma*t
    else:
        boundaries = None
    center_xy = (L / 2.0, L / 2.0)
    xs, dx, ref_grids, bnd, fd_diag = build_reference_geom(p, n_ref, K, L, R0, ETA, T_FINAL, center_xy,
                                                           boundaries=boundaries,
                                                           fd_save_every=o.get("fd_save_every", None))
    dts = np.diff(bnd)            # per-interval dt (VARIABLE when adaptive)
    dt = float(dts.max())         # worker tau-normalization spans the largest interval
    overlap = float(o.get("overlap", 1.5 * ETA))   # OPTIONS-overridable (default 1.5*eta=10.5); B3_DRIVEN_FORCE sets 7.0
    base_dn_margin = float(dn_margin_eta) * ETA
    if box_edges is not None:
        nx = ny = len(box_edges) - 1   # derive grid from explicit edges (center-large geometry)
    boxes = make_boxes(L, nx, ny, overlap, x_edges=box_edges, y_edges=box_edges)
    if box_edges is not None:
        print(f"GEOMETRY: non-uniform core edges {box_edges} -> central core "
              f"[{box_edges[1]},{box_edges[-2]}] (grain in single worker for R<={ (box_edges[-2]-box_edges[1])/2:.1f})")
    print(f"DECOMP B2: {nx}x{ny}={len(boxes)} boxes, overlap={overlap}, n_ref={n_ref}, K={K}, dt={dt}; "
          f"FD R {grain_radius(ref_grids[0], dx):.1f}->{grain_radius(ref_grids[-1], dx):.1f} "
          f"(law {curvature_law_radius(T_FINAL, R0, MU, SIGMA):.1f})")

    periodic_dims = [0, 1] if periodic_embed else None
    multinn = MultiNN(boxes, hidden, out_dim=1, dt=dt, output_activation="sigmoid",
                      periodic_dims=periodic_dims, domain=L, seed=seed)
    loss, set_batch, W, terms = make_loss(multinn, p, [], w_ic=o["w_ic"], w_dn=o["w_dn"],
                                          w_cont=o["w_cont"], w_slope=w_slope, w_trend=w_trend,
                                          pde_project=project, pde_eps=pde_eps, w_profile=w_profile)
    print(f"option {option}: nx={nx} hidden={hidden} w_ic={o['w_ic']} w_dn={o['w_dn']} "
          f"w_cont={o['w_cont']} w_slope={w_slope} w_trend={w_trend} gradnorm={gradnorm} "
          f"n_dn={n_dn} n_sl={n_sl} n_tr={n_tr} lbfgs_pairs={lbfgs_pairs} pde_band={pde_band} "
          f"K={K} adam={adam} lb={lb} dn_margin_eta={dn_margin_eta} "
          f"dn_margin_start_radius={dn_margin_start_radius}")
    outdir = outdir or os.path.join(ROOT, "outputs", f"b2_opt{option}")

    # warm_weights_path: initialize multinn from compatible pre-trained weights before any interval.
    # Architecture must match exactly (same nx, hidden, box_edges → same n_params).
    # For Option B' tail runs: warm-start from B4N_GEOM_CENTER final weights (same 3x3 geometry).
    # This gives the optimizer a good starting basin for the first interval instead of random init.
    warm_weights_path = o.get("warm_weights_path")
    if warm_weights_path is not None and os.path.exists(warm_weights_path):
        w_warm = np.load(warm_weights_path).astype(np.float64)
        n_p = int(sum(int(np.prod(v.shape)) for v in multinn.trainable_variables))
        assert w_warm.shape == (n_p,), (
            f"warm_weights_path: shape {w_warm.shape} != expected ({n_p},)")
        multinn.set_flat_params(w_warm)
        print(f"WARM INIT: loaded {warm_weights_path}  n_params={n_p}")
    elif warm_weights_path is not None:
        print(f"WARM INIT SKIP: {warm_weights_path} not found -> random init")

    # phi_init_path: load a previously extracted PINN handoff field as the IC (unsupervised transfer —
    # the field comes from the PINN itself, not FD labels). R0_phys is re-measured from the loaded field.
    phi_init_path = o.get("phi_init_path")
    if phi_init_path is not None:
        phi_k = np.load(phi_init_path).astype(np.float64)
        assert phi_k.shape == (n_ref, n_ref), (
            f"phi_init_path shape {phi_k.shape} != expected ({n_ref},{n_ref})")
        R0_phys = float(grain_radius(np.clip(phi_k, 0.0, 1.0), dx))
        print(f"HANDOFF IC loaded from {phi_init_path}: R0_meas={R0_phys:.3f} cells  "
              f"phi=[{phi_k.min():.4f},{phi_k.max():.4f}]")
    else:
        phi_k = np.asarray(ref_grids[0])
        R0_phys = float(grain_radius(phi_k, dx))
    phi_prev = None; traj, mses = [phi_k], []
    R_pinn_hist = [R0_phys]; R_ref_hist = [R0_phys]   # PHYSICAL radii, maintained (resume-safe)
    dn_margin_history = []
    loss_history = []  # per-interval {pde,ic,dn,cont (unweighted), w_*, total} for live loss viz
    diag_history = []  # per-interval PINN/ref physical diagnostics (mass, max_phi, energy, projection)
    eta_cells = ETA / dx_print

    # RESUME from a per-interval checkpoint (weights.npy + metrics.json) so a teardown/kill mid-run
    # costs minutes, not the whole horizon. Same option+seed => identical geometry/architecture, so the
    # carried-forward IC is reconstructed by predicting the last completed interval's end field.
    start_k = 0
    if resume:
        mpath = os.path.join(outdir, "metrics.json"); wpath = os.path.join(outdir, "weights.npy")
        if os.path.exists(mpath) and os.path.exists(wpath):
            with open(mpath) as _f:
                prev = json.load(_f)
            sk = int(prev.get("intervals_completed", 0))
            if 0 < sk < K:
                multinn.set_flat_params(np.load(wpath))
                start_k = sk
                mses = list(prev.get("mse_per_interval", []))[:sk]
                loss_history = list(prev.get("loss_terms_per_interval", []))[:sk]
                diag_history = list(prev.get("diag_per_interval", []))[:sk]
                dn_margin_history = list(prev.get("denoise_margin_cells_per_interval", []))[:sk]
                R_pinn_hist = list(prev.get("R_pinn", [R0_phys]))[:sk + 1]
                R_ref_hist = [float(grain_radius(np.asarray(g), dx)) for g in ref_grids[:sk + 1]]
                phi_k = np.asarray(multinn.predict_grid(xs, float(dts[sk - 1])))
                traj = [phi_k]
                print(f"RESUME: continuing from interval {sk}/{K}; restored weights + "
                      f"{len(mses)} scalar records; R_pinn[-1]={R_pinn_hist[-1]/dx:.2f} cells")
            else:
                print(f"RESUME: checkpoint interval {sk} not usable (0 or >=K) -> fresh start")
        else:
            print("RESUME: no checkpoint found -> fresh start")

    def save(done):
        R_pinn = list(R_pinn_hist)
        R_ref = list(R_ref_hist)
        scheme = f"decomposition({nx}x{ny})+time-march"
        if project:
            scheme += "+projected-RHS"
        scheme += "+denoising"
        if base_dn_margin > 0.0:
            scheme += "+distance-margin"
        if periodic_embed:
            scheme += "+periodic-embed"
        ext = extinction_summary(diag_history, eta_cells) if diag_history else {}
        # annular field error: |pred-ref| restricted to the annulus between the two radii (final)
        annular = None
        k_last = len(R_pinn_hist) - 1   # ref_grids index of the latest PINN frame (resume-safe)
        if len(traj) > 1 and k_last >= 1:
            rr = np.sqrt(((np.add.outer(xs, np.zeros_like(xs)) - L / 2.0) ** 2)
                         + ((np.add.outer(np.zeros_like(xs), xs) - L / 2.0) ** 2))
            r_lo = min(R_pinn[-1], R_ref[-1]); r_hi = max(R_pinn[-1], R_ref[-1])
            band = (rr >= r_lo - dx) & (rr <= r_hi + dx)
            if band.any():
                annular = float(np.mean((np.clip(traj[-1], 0, 1)[band]
                                         - np.asarray(ref_grids[k_last])[band]) ** 2))
        m = {"benchmark": "b2_grain_shrinkage", "scheme": scheme, "option": option,
             "quick": quick, "units": units_label, "dx": dx, "mu": MU, "sigma": SIGMA,
             "delta_g": delta_g, "eta": ETA, "eta_cells": eta_cells, "R0": R0,
             "projected_rhs": bool(project), "pde_eps": pde_eps, "pde_bulk_frac": pde_bulk_frac,
             "periodic_embed": bool(periodic_embed),
             "t_ext_sharp": float(R0 ** 2 / (2.0 * MU * SIGMA)),
             "boxes": len(boxes), "K_intervals": K, "dt": dt,
             "fd_dt_stable": fd_diag["fd_dt_stable"], "fd_dt_used": fd_diag["fd_dt_used"],
             "fd_boundary_max_abs_error": fd_diag["fd_boundary_max_abs_error"],
             "fd_save_every": fd_diag["fd_save_every"],
             "box_edges": list(box_edges) if box_edges is not None else None,
             "overlap": overlap, "center_xy": list(center_xy),
             "cycles": cycles, "adam_per_cycle": adam_per_cycle, "loss_thresh": loss_thresh,
             "restore_best_cycle": bool(restore_best_cycle),
             "weights": {"pde": 1.0, "ic": o["w_ic"], "dn": o["w_dn"], "cont": o["w_cont"]},
             "denoise_margin_eta": float(dn_margin_eta), "denoise_margin_cells": base_dn_margin,
             "denoise_margin_start_radius": dn_margin_start_radius,
             "denoise_margin_cells_per_interval": dn_margin_history,
             "hidden": list(hidden), "gradnorm": bool(gradnorm), "n_dn": n_dn,
             "lbfgs_pairs": lbfgs_pairs, "loss_terms_per_interval": loss_history,
             "diag_per_interval": diag_history, "extinction": ext,
             "annular_mse_final": annular,
             "seam": (seam_diagnostics(traj[-1], xs, L, nx, edges=box_edges) if len(traj) > 1 else None),
             "done": done, "intervals_completed": len(mses), "mse_per_interval": mses,
             "mse_vs_reference_mean": (float(np.mean(mses)) if mses else None),
             "R_pinn": R_pinn, "R_ref": R_ref,
             "R_pinn_cells": [r / dx for r in R_pinn], "R_ref_cells": [r / dx for r in R_ref],
             "t_boundaries": bnd.tolist(),
             "radius_mae": (float(np.mean(np.abs(np.array(R_pinn) - np.array(R_ref)))) if mses else None),
             "final_radius_rel_err": (abs(R_pinn[-1] - R_ref[-1]) / R_ref[-1] if mses else None),
             "tf": info["tf_version"], "gpus": info["gpus"]}
        io.save_json(m, os.path.join(outdir, "metrics.json"))
        # CHECKPOINT the ensemble weights every interval (flat params) -> resumable / warm-startable
        try:
            np.save(os.path.join(outdir, "weights.npy"), multinn.get_flat_params().numpy())
        except Exception as e:  # noqa: BLE001
            print("weights checkpoint skipped:", repr(e))
        return m

    pairs_ready = False
    warm_saved = os.path.exists(os.path.join(outdir, "weights_warm.npy"))
    for k in range(start_k, K):
        dt_k = float(dts[k])  # this interval's dt (variable when adaptive; == dt otherwise)
        current_radius = float(grain_radius(np.clip(phi_k, 0, 1), dx))
        dn_margin = base_dn_margin
        if dn_margin_start_radius is not None and current_radius > dn_margin_start_radius:
            dn_margin = 0.0
        elif dn_margin_start_radius is not None and dn_margin_ramp_width:
            # GRADUAL release (Plan A): ramp margin 0 -> base over R in [start-ramp_width, start],
            # instead of a hard 0->base switch, so phi=1 core control is removed smoothly while the
            # projected-RHS + bulk collocation take over bulk control.
            frac = min(1.0, max(0.0, (dn_margin_start_radius - current_radius) / dn_margin_ramp_width))
            dn_margin = base_dn_margin * frac
        dn_margin_history.append(dn_margin)
        # WARM-START checkpoint: snapshot weights the moment R first crosses below the release radius,
        # so future sub-release tail experiments can resume from here and skip the ~known-good early run.
        if (not warm_saved and dn_margin_start_radius is not None
                and current_radius <= dn_margin_start_radius):
            try:
                np.save(os.path.join(outdir, "weights_warm.npy"), multinn.get_flat_params().numpy())
                io.save_json({"interval": k, "radius_cells": current_radius / dx,
                              "R_release": dn_margin_start_radius, "option": option},
                             os.path.join(outdir, "warm_meta.json"))
                print(f"  WARM CHECKPOINT saved at interval {k}, R={current_radius/dx:.2f} cells "
                      f"(reuse via weights_warm.npy)")
            except Exception as e:  # noqa: BLE001
                print("warm checkpoint skipped:", repr(e))
            warm_saved = True
        # SLOPE target = curvature velocity (PDE RHS) on the carried-forward IC, in interface band
        v_target = dphi_dt(np.clip(phi_k, 0.0, 1.0), p, dx) if n_sl > 0 else None
        # TREND target at tau=dt: interval 1 FD-seeded; later = extrapolate the prev achieved step
        trend_target = None
        if n_tr > 0:
            if k == 0:
                trend_target = np.asarray(ref_grids[1])  # bootstrap interval 1 (1 of K)
            elif phi_prev is not None:
                trend_target = np.clip(2.0 * phi_k - phi_prev, 0.0, 1.0)
        batch, pairs = build_batch(multinn, phi_k, xs, dt_k, n_f, n_ic, n_dn, n_cont, ETA, seed + 1 + k,
                                   dn_margin=dn_margin, v_target=v_target, n_sl=n_sl,
                                   trend_target=trend_target, n_tr=n_tr, pde_band=pde_band,
                                   pde_bulk_frac=pde_bulk_frac)
        # rebuild loss with the actual neighbour pairs (continuity) on the first EXECUTED interval
        # (k==0 fresh, or k==start_k on resume)
        if not pairs_ready:
            pairs_ready = True
            loss, set_batch, W, terms = make_loss(multinn, p, pairs, w_ic=o["w_ic"], w_dn=o["w_dn"],
                                                  w_cont=o["w_cont"], w_slope=w_slope,
                                                  pde_project=project, pde_eps=pde_eps, w_profile=w_profile)
        set_batch(batch)
        decomp_fit_interval(multinn, loss, adam, lb, lr=1e-3, max_restarts=1,
                            gradnorm=gradnorm, terms=terms, W=W,
                            num_correction_pairs=lbfgs_pairs,
                            cycles=cycles, adam_per_cycle=adam_per_cycle,
                            loss_thresh=loss_thresh, log_cycles=True,
                            restore_best_cycle=restore_best_cycle)
        _, ld = loss()  # post-fit per-term (unweighted) losses + current (GradNorm) weights
        loss_history.append({"interval": k + 1, "total": float(ld["total"]),
                             "pde": float(ld["pde"]), "ic": float(ld["ic"]),
                             "dn": float(ld["dn"]), "cont": float(ld["cont"]),
                             "slope": float(ld["slope"]), "trend": float(ld["trend"]),
                             "w_pde": float(W["pde"].numpy()), "w_ic": float(W["ic"].numpy()),
                             "w_dn": float(W["dn"].numpy()), "w_cont": float(W["cont"].numpy()),
                             "w_slope": float(W["slope"].numpy()), "w_trend": float(W["trend"].numpy())})
        phi_next = np.asarray(multinn.predict_grid(xs, dt_k))
        mse = float(np.mean((phi_next - np.asarray(ref_grids[k + 1])) ** 2))
        traj.append(phi_next); mses.append(mse)
        # save_phi_every: persist phi field every N intervals for post-hoc snapshot figures
        _save_phi_every = o.get("save_phi_every", 0)
        if _save_phi_every > 0 and (k + 1) % _save_phi_every == 0:
            _sdir = os.path.join(outdir, "phi_snapshots")
            os.makedirs(_sdir, exist_ok=True)
            _R_snap = float(grain_radius(np.clip(phi_next, 0.0, 1.0), dx))
            np.save(os.path.join(_sdir, f"phi_iv{k+1:04d}_R{_R_snap:.2f}.npy"), phi_next)
        R_pinn_hist.append(float(grain_radius(np.clip(phi_next, 0, 1), dx)))
        R_ref_hist.append(float(grain_radius(np.asarray(ref_grids[k + 1]), dx)))
        # physical diagnostics (mass/max_phi/free-energy/projection) on PINN + reference fields
        dp = diagnose_frame(np.clip(phi_next, 0.0, 1.0), p, dx, eps=pde_eps)
        dr = diagnose_frame(np.asarray(ref_grids[k + 1]), p, dx, eps=pde_eps)
        diag_history.append({
            "interval": k + 1, "t": float(bnd[k + 1]),
            "pinn": {"R_cells": dp["radius"] / dx, "mass": dp["mass"], "max_phi": dp["max_phi"],
                     "free_energy": dp["free_energy"], "interface": dp["interface_present"],
                     "proj_frac_total": dp["projection"]["frac_total"],
                     "proj_frac_interface": dp["projection"]["frac_interface"],
                     "proj_frac_bulk_low": dp["projection"]["frac_bulk_low"],
                     "proj_frac_bulk_high": dp["projection"]["frac_bulk_high"]},
            "ref": {"R_cells": dr["radius"] / dx, "mass": dr["mass"], "max_phi": dr["max_phi"],
                    "free_energy": dr["free_energy"], "interface": dr["interface_present"]}})
        m = save(False)
        print(f"  [interval {k + 1}/{K}] t={bnd[k + 1]:.1f} mse_vs_ref={mse:.3e} "
              f"R_pinn={dp['radius']/dx:.2f} R_ref={dr['radius']/dx:.2f} (cells) "
              f"maxphi={dp['max_phi']:.3f} mass={dp['mass']:.3g} dF={dp['free_energy']-dr['free_energy']:+.3g} "
              f"proj_if={dp['projection']['frac_interface']:.2f} dn_margin={dn_margin:.2f}")
        # --handoff-at N: save the PINN field at end of interval N as a transferred IC for another run
        if handoff_at is not None and (k + 1) == handoff_at:
            hdir = os.path.join(outdir, "handoff")
            os.makedirs(hdir, exist_ok=True)
            np.save(os.path.join(hdir, "handoff_phi.npy"), phi_next)
            import json as _json
            _json.dump({"source_option": option, "interval": k + 1,
                        "t_end": float(bnd[k + 1]), "warm_tau": float(dts[k]),
                        "R_pinn_cells": float(dp["radius"] / dx),
                        "max_phi": float(dp["max_phi"])},
                       open(os.path.join(hdir, "handoff_meta.json"), "w"), indent=2)
            print(f"HANDOFF saved at interval {k+1}/{K}  R_pinn={dp['radius']/dx:.3f}  "
                  f"t={bnd[k+1]:.2f}  -> {hdir}")
            return  # exit cleanly after saving; --resume picks up from weights.npy if needed
        phi_prev = phi_k; phi_k = phi_next

    m = save(True)
    try:
        R_pinn = m["R_pinn"]; R_ref = m["R_ref"]
        viz_all(traj, ref_grids, xs, bnd, R_pinn, R_ref, boxes, outdir)
        print("viz saved ->", outdir)
    except Exception as e:  # noqa: BLE001
        print("viz skipped:", repr(e))
    ex = m.get("extinction", {}) or {}
    print(f"\n=== B2 DECOMP [{option}]: mean MSE_vs_ref={m['mse_vs_reference_mean']:.3e} "
          f"final_radius_rel_err={m['final_radius_rel_err']:.3f} ===")
    print(f"    final R_pinn={m['R_pinn'][-1]:.3g} R_ref={m['R_ref'][-1]:.3g} (phys)  "
          f"= {m['R_pinn_cells'][-1]:.2f}/{m['R_ref_cells'][-1]:.2f} cells  "
          f"final max_phi(pinn)={diag_history[-1]['pinn']['max_phi']:.3f} "
          f"mass(pinn)={diag_history[-1]['pinn']['mass']:.3g}")
    print(f"    extinction t: pinn={ex.get('extinction_t_pinn')} ref={ex.get('extinction_t_ref')}  "
          f"clearance t (max_phi<0.5): pinn={ex.get('clearance_t_pinn')} ref={ex.get('clearance_t_ref')}  "
          f"R^2-slope(>2.5eta): pinn={ex.get('R2_slope_pinn')} ref={ex.get('R2_slope_ref')}")
    if m.get("seam"):
        print(f"    seam_max_jump={m['seam']['seam_max_jump']:.3g} (baseline {m['seam']['baseline_max_jump']:.3g})  "
              f"annular_mse_final={m.get('annular_mse_final')}")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--option", default="A", choices=sorted(OPTIONS))
    ap.add_argument("--dn-margin-eta", type=float, default=None,
                    help="Override denoising distance margin as a multiple of eta")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from <outdir>/weights.npy + metrics.json (per-interval checkpoint)")
    ap.add_argument("--handoff-at", type=int, default=None,
                    help="Save PINN field after interval N to <outdir>/handoff/ and exit (IC transfer)")
    ap.add_argument("--outdir", default=None); a = ap.parse_args()
    run(quick=a.quick, outdir=a.outdir, option=a.option, dn_margin_eta=a.dn_margin_eta,
        resume=a.resume, handoff_at=a.handoff_at)
