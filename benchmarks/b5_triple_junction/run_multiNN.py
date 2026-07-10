"""Benchmark 5 - triple junction, MultiNN/time-marched PINNs-MPF.

Architecture: Modernized MultiNN — one 4-output (logit) softmax worker per spatial batch.
  4 workers on a 2×2 spatial decomposition of [0,1]², overlap=0.1.
  Softmax output enforces sum(phi)=1 by construction.

Reference: validated_benchmarks/B5_triple_junction/reference/b5_ref_90_to_120_128.npz
  N=128, L=1, eta=6/64, mu=1e-4, sigma=1, tau_c=87.89, t_final=200 (2.28 tau_c).
  IC: periodic-safe four-phase six-junction geometry.

Time windows: 8 windows of dt=25 time units.

IC handoff modes (controlled by --rollout flag):
  default (FD handoff): IC at every window start taken from the FD reference frame.
    Avoids error accumulation; used for per-window physics-loss validation.
  --rollout: first window IC from FD; subsequent windows use the PINN prediction
    at the end of the previous window. Tests PINN auto-regressive stability.
  In both modes evaluation always uses the reference grid so phi_pinn and phi_fd
  are the same shape for unambiguous MSE comparison.

Discrete loss activation:
  Validated default: all loss terms active from window 1
    (PDE + IC + bulk stabilization + continuity + periodic seam).
  Optional --staged-loss mode:
    Window 1: PDE + IC only; windows 2+: all terms active.

Usage:
  python benchmarks/b5_triple_junction/run_multiNN.py [--quick] [--outdir PATH]
  python benchmarks/b5_triple_junction/run_multiNN.py --quick    # smoke ladder: tiny net, 2 windows

  # Validated 128x128 production configuration:
  python benchmarks/b5_triple_junction/run_multiNN.py \\
    --outdir runs/b5_triple_junction_128/production
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time as _time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pinns_mpf as pm                                            # noqa: E402
from pinns_mpf import io                                          # noqa: E402
from pinns_mpf.config import PhysicalParams                       # noqa: E402
from pinns_mpf.decomposition.decomp import (                      # noqa: E402
    MultiNN, make_boxes, neighbor_pairs, decomp_fit_interval,
)
from pinns_mpf.decomposition.decomp_multiphase import (           # noqa: E402
    build_batch_multiphase, make_loss_multiphase, predict_grid_multiphase,
)

# ── reference path ────────────────────────────────────────────────────────────

REF_NPZ = os.path.join(
    ROOT,
    "validated_benchmarks",
    "B5_triple_junction",
    "reference",
    "b5_ref_90_to_120_128.npz",
)

# ── B5 physics ────────────────────────────────────────────────────────────────

L       = 1.0
ETA     = 6.0 / 64.0          # 0.09375  (eta=6dx)
MU      = 1e-4
SIGMA   = 1.0
T_FINAL = 200.0
TAU_C   = ETA**2 / (MU * SIGMA)  # 87.89

N_WINDOWS    = 8
DT_WINDOW    = T_FINAL / N_WINDOWS  # 25.0 per window
N_SPATIAL    = 2                    # 2×2 decomposition
OVERLAP      = 0.10                 # physical units (10% of L)
NPHASES      = 4


# ── FD reference loading ──────────────────────────────────────────────────────

def load_fd_reference():
    d = np.load(REF_NPZ)
    return d["phi"], d["times"]          # [T, 4, N, N], [T]


def fd_frame_at(phi_fd, times_fd, t_target):
    """Return the FD frame closest to t_target."""
    idx = int(np.argmin(np.abs(times_fd - t_target)))
    return phi_fd[idx]                   # [4, N, N]


# ── architecture ──────────────────────────────────────────────────────────────

def build_multinn(dt_window, hidden=(128,)*6, seed=1234):
    boxes = make_boxes(L, N_SPATIAL, N_SPATIAL, OVERLAP)
    model = MultiNN(boxes, hidden=hidden, out_dim=NPHASES, dt=dt_window,
                    output_activation="linear", periodic_dims=[0, 1],
                    domain=L, seed=seed)
    return model, boxes


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(multinn, xs_fd, phi_fd_frame, dt_w):
    """Field MSE, argmax disagreement, sum error against one FD frame.

    Always evaluates on xs_fd (the FD reference grid) so phi_pinn and
    phi_fd_frame have the same shape regardless of the --quick flag.

    xs_fd        : 1-D array of N spatial coords (the FD reference grid)
    phi_fd_frame : [4, N, N]
    dt_w         : window duration (predict at tau=dt_w, end of local window)
    """
    phi_pinn = predict_grid_multiphase(multinn, xs_fd, dt_w)  # [4, N, N]
    phi_fd   = phi_fd_frame                                    # [4, N, N]

    mse     = float(np.mean((phi_pinn - phi_fd)**2))
    mae     = float(np.mean(np.abs(phi_pinn - phi_fd)))
    argmax_pinn = phi_pinn.argmax(axis=0)
    argmax_fd   = phi_fd.argmax(axis=0)
    argmax_disagree = float((argmax_pinn != argmax_fd).mean())
    sum_err = float(np.abs(phi_pinn.sum(axis=0) - 1.0).max())

    return phi_pinn, {
        "mse_vs_fd":          mse,
        "mae_vs_fd":          mae,
        "argmax_disagree":    argmax_disagree,
        "max_sum_err":        sum_err,
    }


def seam_diagnostics(phi_pinn):
    """Measure periodic wrap-seam mismatch in the PINN field.

    The FD reference uses periodic BCs, so phi[x=0] == phi[x=L] and
    phi[y=0] == phi[y=L]. The MLP uses Fourier (sin/cos) embedding which
    is exactly periodic by construction, but each worker covers only a
    sub-domain. Seam mismatch between Box 0/1 (left/right) and Box 0/2
    (bottom/top) continuity is already penalised via the continuity loss,
    but the OUTER wrap seams (x=0 edge of Box 0 vs x=1 edge of Box 1, and
    y=0 edge of Box 0 vs y=1 edge of Box 2) are NOT in any overlap halo
    and may show artifacts.

    phi_pinn : [4, N, N]  float64
    Returns dict with max and mean absolute seam errors across all 4 phases.
    """
    # Left column (ix=0) vs right column (ix=-1): wrap in x
    lr_err = np.abs(phi_pinn[:, 0, :] - phi_pinn[:, -1, :])   # [4, N]
    # Bottom row (iy=0) vs top row (iy=-1): wrap in y
    bt_err = np.abs(phi_pinn[:, :, 0] - phi_pinn[:, :, -1])   # [4, N]
    return {
        "seam_lr_max":  float(lr_err.max()),
        "seam_lr_mean": float(lr_err.mean()),
        "seam_bt_max":  float(bt_err.max()),
        "seam_bt_mean": float(bt_err.mean()),
    }


def angle_diagnostics(phi_field):
    """Multi-junction angle RMS using the article reference detector.

    phi_field : [4, N, N]
    Returns stats dict or None if import fails.
    """
    try:
        sys.path.insert(0, os.path.join(ROOT, "reference"))
        from b5_junction_analysis import junction_angles_all
        _, stats = junction_angles_all(phi_field, L=L)
        return stats
    except Exception:
        return None


# ── training loop ─────────────────────────────────────────────────────────────

def run(quick=False, outdir=None, rollout=False, seed=1234,
        override_adam_steps=None, override_lbfgs_iters=None,
        override_hidden_width=None, override_hidden_depth=None,
        override_n_f=None, override_pbc_weight=None,
        override_n_windows=None,
        override_pde_weight=None, override_ic_weight=None,
        override_full_loss_from_start=None,
        override_ref_npz=None,
        override_start_window=None,
        override_load_weights=None):
    import tensorflow as tf

    if quick:
        # smoke ladder: tiny net, 2 windows, minimal collocation
        hidden         = (32, 32)
        n_f, n_ic, n_dn, n_cont = 600, 400, 200, 100
        adam_steps     = 100
        lbfgs_iters    = 0
        n_lbfgs_rounds = 0
        n_win          = 2
        dt_w           = T_FINAL / 2.0   # 2 windows of dt=100 each
        pde_chunk      = None            # tiny net fits in memory without chunking
        pde_eager      = False           # @tf.function fine for the tiny smoke net
    else:
        # Validated 128x128 production setting matching the packaged B5 summary:
        # 8 windows of dt=25, hidden=(64,)*6, Adam-only, full loss from window 1.
        hidden         = (64,) * 6
        n_f, n_ic, n_dn, n_cont = 4800, 1500, 800, 400
        adam_steps     = 800
        lbfgs_iters    = 0
        n_lbfgs_rounds = 1
        n_win          = N_WINDOWS
        dt_w           = DT_WINDOW       # 8 windows of dt=25 each
        pde_chunk      = None
        pde_eager      = False

    # Apply CLI overrides (None = keep the quick/prod default above).
    if override_hidden_width is not None or override_hidden_depth is not None:
        w = override_hidden_width if override_hidden_width is not None else hidden[0]
        d = override_hidden_depth if override_hidden_depth is not None else len(hidden)
        hidden = (w,) * d
    if override_adam_steps  is not None: adam_steps  = override_adam_steps
    if override_lbfgs_iters is not None: lbfgs_iters = override_lbfgs_iters
    if override_n_f         is not None: n_f         = override_n_f
    pbc_weight_active = 1.0 if override_pbc_weight is None else float(override_pbc_weight)
    if override_n_windows    is not None: n_win    = override_n_windows
    w_pde = 1.0 if override_pde_weight is None else float(override_pde_weight)
    w_ic  = 2.0 if override_ic_weight  is None else float(override_ic_weight)
    if override_full_loss_from_start is None:
        full_loss_from_start = not quick
    else:
        full_loss_from_start = bool(override_full_loss_from_start)
    ref_npz_path  = override_ref_npz if override_ref_npz is not None else REF_NPZ
    win_offset    = 0 if override_start_window is None else int(override_start_window) - 1
    load_weights  = override_load_weights   # path to weights.npz, or None

    # Evaluation uses the FD reference grid at its native resolution (e.g. 64×64 or
    # 128×128), determined by the loaded reference NPZ dimensions — so phi_pinn and
    # phi_fd are always the same shape for unambiguous comparison.

    info = pm.setup(seed=seed)
    p    = PhysicalParams(mu=MU, sigma=SIGMA, eta=ETA)
    outdir = outdir or os.path.join(ROOT, "runs", "b5_triple_junction_128",
                                     "quick" if quick else "production")
    os.makedirs(outdir, exist_ok=True)

    actual_t_final = n_win * dt_w   # true simulated horizon (T_FINAL is a doc constant)
    print(f"[B5 MultiNN] {'QUICK' if quick else 'PRODUCTION'} | "
          f"n_win={n_win}  dt_w={dt_w}  hidden={hidden}  seed={seed}")
    print(f"  Reference: {ref_npz_path}")
    print(f"  Output:    {outdir}")
    print(f"  tau_c={TAU_C:.2f}  t_final={actual_t_final}  ({actual_t_final/TAU_C:.2f} tau_c)")
    print(f"  env: {info}")

    # Load FD reference — xs_fd is the single eval/batch grid used throughout
    _ref = np.load(ref_npz_path)
    phi_fd, times_fd = _ref["phi"], _ref["times"]   # [T, 4, N, N], [T]
    xs_fd = np.linspace(0, L, phi_fd.shape[-1], endpoint=False)  # N pts (reference-native)

    # Guard: the reference must cover the full simulated horizon so fd_frame_at()
    # interpolates (never extrapolates) at every window endpoint. Catches loading a
    # too-short or wrong reference before burning GPU hours.
    if times_fd[-1] < actual_t_final - 1e-6:
        raise ValueError(
            f"Reference {os.path.basename(ref_npz_path)} spans t=[0,{times_fd[-1]:.1f}] but the "
            f"run needs t=[0,{actual_t_final:.1f}] ({n_win}×{dt_w}). Wrong or too-short reference.")
    print(f"  Reference grid: {phi_fd.shape[-1]}×{phi_fd.shape[-1]}  "
          f"frames={len(times_fd)}  span=t[0,{times_fd[-1]:.1f}]")

    # Build architecture
    multinn, boxes = build_multinn(dt_w, hidden=hidden, seed=seed)
    n_params = sum(int(np.prod(v.shape)) for v in multinn.trainable_variables)
    print(f"  Workers: {len(multinn.workers)}  |  total params: {n_params}")
    print(f"  IC mode: {'ROLLOUT (PINN carries forward after win 1)' if rollout else 'FD handoff each window'}")

    # Warm-start: load weights from a prior checkpoint before any window training
    if load_weights is not None:
        io.checkpoint.load_weights(multinn, load_weights)
        print(f"  Warm-start weights loaded from: {load_weights}")

    pairs = neighbor_pairs(boxes, N_SPATIAL, N_SPATIAL)
    (loss_fn, set_batch, W, terms, nonpde_loss_fn,
     pde_grads_fn, terms_per_worker_fn) = make_loss_multiphase(
        multinn, p, pairs, w_pde=w_pde, w_ic=w_ic, w_dn=0.0, w_cont=0.0, w_pbc=0.0,
        pde_chunk=pde_chunk)

    all_metrics = []
    t_wall_0 = _time.time()

    # phi_ic tracks the IC field for the current window.
    # FD handoff (default): always loaded fresh from FD at t_start.
    # Rollout: first window from FD, then carried from previous PINN prediction.
    # Both modes: phi_ic is [4, N, N], aligned with xs_fd.
    phi_ic = fd_frame_at(phi_fd, times_fd, 0.0)   # initialize; overwritten in loop

    for loop_idx in range(n_win):
        win_idx = win_offset + loop_idx   # absolute window index (0-based)
        t_start = win_idx * dt_w
        t_end   = t_start + dt_w

        # Determine IC for this window
        if not rollout or loop_idx == 0:
            # FD handoff: load the FD frame closest to t_start
            phi_ic = fd_frame_at(phi_fd, times_fd, t_start)  # [4, N, N]
        # else: phi_ic already set to the PINN prediction from the previous window

        print(f"\n[win {win_idx+1}/{n_win + win_offset}]  t={t_start:.1f}->{t_end:.1f}"
              f"  ({t_start/TAU_C:.3f}->{t_end/TAU_C:.3f} tau_c)"
              f"  IC={'PINN rollout' if rollout and loop_idx > 0 else 'FD frame'}")

        # Discrete activation: validated default uses all terms from window 1.
        if loop_idx == 0 and not full_loss_from_start:
            W["dn"].assign(0.0); W["cont"].assign(0.0); W["pbc"].assign(0.0)
            print("  Losses: PDE + IC only (first window)")
        else:
            W["dn"].assign(1.0); W["cont"].assign(1.0); W["pbc"].assign(pbc_weight_active)
            print("  Losses: PDE + IC + bulk stabilization + continuity + PBC seam")

        # Build batch on the FD grid (xs_fd, reference-native N x N)
        batch, _ = build_batch_multiphase(
            multinn, phi_ic, xs_fd, dt_w,
            n_f, n_ic, n_dn, n_cont, ETA,
            seed=seed + loop_idx)
        set_batch(batch)

        # Train this window
        decomp_fit_interval(
            multinn, loss_fn,
            adam_steps, lbfgs_iters,
            lr=1e-3,
            max_restarts=max(1, n_lbfgs_rounds),
            num_correction_pairs=20 if quick else 50,
            pde_eager=pde_eager,
            nonpde_loss=nonpde_loss_fn if pde_eager else None,
            pde_grads_fn=pde_grads_fn if pde_eager else None)

        # Evaluate on xs_fd (reference-native grid) — avoids grid mismatch
        phi_fd_end        = fd_frame_at(phi_fd, times_fd, t_end)  # [4, N, N]
        phi_pinn_end, metrics = evaluate(multinn, xs_fd, phi_fd_end, dt_w)
        metrics["window"]  = int(win_idx + 1)
        metrics["t_start"] = float(t_start)
        metrics["t_end"]   = float(t_end)
        metrics["t_tau_c"] = float(t_end / TAU_C)
        metrics["wall_s"]  = round(_time.time() - t_wall_0, 1)
        metrics["ic_mode"] = "rollout" if (rollout and loop_idx > 0) else "fd_handoff"

        # Final per-term loss breakdown (unweighted, for diagnostics)
        final_terms = terms()
        metrics["pde_loss"]  = float(final_terms["pde"].numpy())
        metrics["ic_loss"]   = float(final_terms["ic"].numpy())
        metrics["dn_loss"]   = float(final_terms["dn"].numpy())
        metrics["cont_loss"] = float(final_terms["cont"].numpy())
        metrics["pbc_loss"]  = float(final_terms["pbc"].numpy())

        # Per-box loss breakdown (for pyramid threshold calibration T = 5× per-box PDE)
        pw = terms_per_worker_fn()
        metrics["pde_per_worker"] = pw["pde_per_worker"]
        metrics["ic_per_worker"]  = pw["ic_per_worker"]

        # Periodic-boundary seam mismatch diagnostic
        seam = seam_diagnostics(phi_pinn_end)
        metrics.update(seam)

        # Multi-junction angle diagnostics on the reference-native PINN field
        angle_stats = angle_diagnostics(phi_pinn_end)
        if angle_stats:
            metrics["n_junctions"] = angle_stats["n_detected"]
            metrics["rms_dev_120"] = round(angle_stats["rms_dev_from_120"], 2)

        print(f"  MSE={metrics['mse_vs_fd']:.3e}  argmax_disagree={metrics['argmax_disagree']:.3f}"
              f"  sum_err={metrics['max_sum_err']:.2e}"
              f"  seam_lr={seam['seam_lr_max']:.2e}  seam_bt={seam['seam_bt_max']:.2e}"
              f"  pbc_loss={metrics['pbc_loss']:.3e}"
              + (f"  RMS_angle={metrics.get('rms_dev_120','N/A')}°" if "rms_dev_120" in metrics else ""))

        all_metrics.append(metrics)

        # Per-window checkpoint
        ckpt_dir = os.path.join(outdir, f"window_{win_idx+1:02d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        io.checkpoint.save_weights(multinn, os.path.join(ckpt_dir, "weights"))
        io.save_json(metrics, os.path.join(ckpt_dir, "metrics.json"))

        # Rollout: carry the PINN prediction as IC for the next window
        if rollout:
            phi_ic = phi_pinn_end  # [4, N, N] — aligned with xs_fd

    # Save run summary
    summary = {
        "benchmark": "B5_triple_junction_multiNN",
        "reference": os.path.basename(ref_npz_path),
        "quick": quick,
        "n_windows": n_win,
        "dt_window": dt_w,
        "hidden": list(hidden),
        "adam_steps": adam_steps,
        "lbfgs_iters": lbfgs_iters,
        "n_f": n_f,
        "w_pde": w_pde,
        "w_ic": w_ic,
        "loss_schedule": "full_from_start" if full_loss_from_start else "default_w1_pde_ic_only",
        "n_params": n_params,
        "n_workers": len(multinn.workers),
        "rollout_mode": rollout,
        "pde_eager": pde_eager,
        "physics": {"L": L, "eta": ETA, "mu": MU, "sigma": SIGMA,
                    "tau_c": TAU_C, "t_final": actual_t_final},
        "windows": all_metrics,
        "final_mse": all_metrics[-1]["mse_vs_fd"],
        "final_argmax_disagree": all_metrics[-1]["argmax_disagree"],
        "total_wall_s": round(_time.time() - t_wall_0, 1),
    }
    io.save_json(summary, os.path.join(outdir, "summary.json"))
    print(f"\n[B5 MultiNN] Done. Summary -> {outdir}/summary.json")
    print(f"  Final MSE={summary['final_mse']:.3e}"
          f"  argmax_disagree={summary['final_argmax_disagree']:.3f}"
          f"  wall={summary['total_wall_s']}s")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick",         action="store_true")
    ap.add_argument("--rollout",       action="store_true",
                    help="Use PINN prediction as IC for next window instead of FD handoff")
    ap.add_argument("--outdir",        default=None)
    ap.add_argument("--seed",          type=int,   default=1234)
    # Production knob overrides (None = use the quick/prod block defaults)
    ap.add_argument("--adam-steps",    type=int,   default=None, dest="adam_steps",
                    help="Override Adam steps per window")
    ap.add_argument("--lbfgs-iters",   type=int,   default=None, dest="lbfgs_iters",
                    help="L-BFGS iterations per window after Adam (0=disabled)")
    ap.add_argument("--hidden-width",  type=int,   default=None, dest="hidden_width",
                    help="Override hidden layer width (all layers same)")
    ap.add_argument("--hidden-depth",  type=int,   default=None, dest="hidden_depth",
                    help="Override number of hidden layers")
    ap.add_argument("--n-f",           type=int,   default=None, dest="n_f",
                    help="Override PDE collocation points")
    ap.add_argument("--pbc-weight",    type=float, default=None, dest="pbc_weight",
                    help="Override PBC seam loss weight when active (default 1.0)")
    ap.add_argument("--n-windows",     type=int,   default=None, dest="n_windows",
                    help="Override number of windows (production default 8); keeps dt_window=25")
    ap.add_argument("--pde-weight",    type=float, default=None, dest="pde_weight",
                    help="Override PDE loss weight w_pde (default 1.0)")
    ap.add_argument("--ic-weight",     type=float, default=None, dest="ic_weight",
                    help="Override IC loss weight w_ic (default 2.0)")
    loss_group = ap.add_mutually_exclusive_group()
    loss_group.add_argument("--full-loss-from-start", action="store_true",
                            default=None, dest="full_loss_from_start",
                            help="Activate all losses (dn, cont, pbc) from window 1")
    loss_group.add_argument("--staged-loss", action="store_false",
                            dest="full_loss_from_start",
                            help="Use PDE+IC only in window 1, then activate all losses")
    ap.add_argument("--ref-npz",      type=str,   default=None, dest="ref_npz",
                    help="Override FD reference .npz (default: built-in 128x128 B5 reference)")
    ap.add_argument("--start-window", type=int,   default=None, dest="start_window",
                    help="First absolute window number (1-based); sets time offset = (N-1)*dt_w")
    ap.add_argument("--load-weights", type=str,   default=None, dest="load_weights",
                    help="Path to weights.npz checkpoint to warm-start from")
    a = ap.parse_args()
    run(quick=a.quick, outdir=a.outdir, rollout=a.rollout, seed=a.seed,
        override_adam_steps=a.adam_steps,
        override_lbfgs_iters=a.lbfgs_iters,
        override_hidden_width=a.hidden_width,
        override_hidden_depth=a.hidden_depth,
        override_n_f=a.n_f,
        override_pbc_weight=a.pbc_weight,
        override_n_windows=a.n_windows,
        override_pde_weight=a.pde_weight,
        override_ic_weight=a.ic_weight,
        override_full_loss_from_start=a.full_loss_from_start,
        override_ref_npz=a.ref_npz,
        override_start_window=a.start_window,
        override_load_weights=a.load_weights)
