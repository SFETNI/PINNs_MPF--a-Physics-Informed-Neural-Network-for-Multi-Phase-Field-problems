"""Benchmark 2 — 2D grain shrinkage (single NN), dimensional grid-unit config,
TIME-MARCHED (PINNs-MPF discrete-time scheme).

Grid units (PINN-Phase Table 6): Dx=1 (=1.5e-5 m), 128^2 -> [0,128]; sigma=1, mu=1,
eta=10*Dx, Dg=0 (curvature-driven); PERIODIC via the MLP sin/cos embedding.
[0,T] is solved as K short intervals of length dt with a MOVING IC (each interval's
IC = the previous interval's prediction); local time tau in [0,dt]. This is the
original framework's core and is what makes long-horizon coarsening tractable
(full-domain training over the whole horizon fails the causality test).

Validated against the FD reference (reference.pf_solver) + curvature law R^2=R0^2-2 mu sigma t.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pinns_mpf as pm  # noqa: E402
from pinns_mpf import io, viz  # noqa: E402
from pinns_mpf.losses.single_phase_2d import loss_ic, loss_pde  # noqa: E402
from pinns_mpf.training import time_march  # noqa: E402
from reference.pf_solver import (PFParams, curvature_law_radius, grain_radius,  # noqa: E402
                                 init_circle, simulate, stable_dt)

L, R0, ETA, MU, SIGMA, T_FINAL = 128.0, 38.0, 10.0, 1.0, 1.0, 400.0
CENTER = (L / 2, L / 2)


def b2_loss(model, batch, p, weights):
    # PDE enforced only on interface-band collocation; IC at tau=0; DENOISING pins the
    # bulk (phi=1 grain core / phi=0 matrix) far from the interface for all tau -- this is
    # what keeps the sigmoid PINN from "growing" the grain to relieve the bulk PDE residual.
    l_pde = loss_pde(model, batch["X_f"], p)
    l_ic = loss_ic(model, batch["X_ic"], batch["phi_ic"])
    l_dn = loss_ic(model, batch["X_dn"], batch["phi_dn"])  # denoising = data on DEEP bulk only
    # Middle balance: 10x under-shrinks, 2x grows -> 5x targets ratio ~1 (with smaller dt):
    total = l_pde + 5.0 * l_ic + 5.0 * l_dn
    return total, {"total": total, "pde": l_pde, "ic": l_ic, "dn": l_dn}


def build_reference(p, n_ref, n_intervals):
    dx = L / n_ref
    phi0, xs = init_circle(n_ref, L, R0, p.eta, center=CENTER)
    dt = stable_dt(p, dx)
    times, frames, _ = simulate(phi0, p, dx, dt=dt, t_final=T_FINAL,
                                save_every=max(1, int(T_FINAL / dt / (4 * n_intervals))))
    bnd = np.linspace(0, T_FINAL, n_intervals + 1)
    ref_grids = [frames[int(np.argmin(np.abs(times - tb)))] for tb in bnd]
    return xs, dx, ref_grids, bnd


def run(quick=False, outdir=None, seed=1234):
    import tensorflow as tf
    if quick:
        n_ref, K, n_f, n_ic, n_dn, adam_steps, lbfgs_iters = 48, 12, 1500, 400, 400, 150, 120
    else:
        n_ref, K, n_f, n_ic, n_dn, adam_steps, lbfgs_iters = 128, 24, 3000, 1000, 1000, 150, 130

    info = pm.setup(seed=seed)
    print("env:", info)
    p = pm.PhysicalParams(mu=MU, sigma=SIGMA, eta=ETA, delta_g=0.0)
    xs, dx, ref_grids, bnd = build_reference(p, n_ref, K)
    dt = T_FINAL / K
    N = len(xs)
    print(f"grid-unit B2 time-marched: L={L} Dx={dx} eta={ETA} R0={R0} T={T_FINAL}, "
          f"K={K} intervals dt={dt}; FD R {grain_radius(ref_grids[0], dx):.1f}->"
          f"{grain_radius(ref_grids[-1], dx):.1f} (law {curvature_law_radius(T_FINAL, R0, MU, SIGMA):.1f})")

    model = pm.MLP([3, 64, 64, 64, 64, 1], lb=[0, 0, 0], ub=[L, L, dt],  # LOCAL time in [0,dt]
                   output_activation="sigmoid", periodic_dims=[0, 1], seed=seed)
    Xg, Yg = np.meshgrid(xs, xs, indexing="ij")
    flat = np.column_stack([Xg.ravel(), Yg.ravel()])

    def make_batch(phi_k, k):
        rng = np.random.default_rng(seed + 1 + k)
        ii, jj = np.where((phi_k > 0.02) & (phi_k < 0.98))   # interface cells of the moving IC
        has = ii.size > 0
        # IC at tau=0 (interface-concentrated + some uniform)
        n1 = int(0.7 * n_ic)
        if has:
            s = rng.integers(0, ii.size, n1); ic_i, ic_j = ii[s], jj[s]
        else:
            ic_i = rng.integers(0, N, n1); ic_j = rng.integers(0, N, n1)
        su = rng.integers(0, N * N, n_ic - n1); ui, uj = np.unravel_index(su, (N, N))
        ic_i = np.concatenate([ic_i, ui]); ic_j = np.concatenate([ic_j, uj])
        X_ic = np.column_stack([xs[ic_i], xs[ic_j], np.zeros(ic_i.size)])
        phi_ic = phi_k[ic_i, ic_j].reshape(-1, 1)
        # PDE collocation ONLY in the interface band (not the bulk), local time tau in [0,dt]
        if has:
            sb = rng.integers(0, ii.size, n_f)
            xb = np.clip(xs[ii[sb]] + rng.uniform(-1.5 * ETA, 1.5 * ETA, n_f), 0, L)
            yb = np.clip(xs[jj[sb]] + rng.uniform(-1.5 * ETA, 1.5 * ETA, n_f), 0, L)
        else:
            xb, yb = rng.uniform(0, L, n_f), rng.uniform(0, L, n_f)
        X_f = np.column_stack([xb, yb, rng.uniform(0, dt, n_f)])
        # DENOISING: only DEEP bulk (far from the moving interface so pinning won't block
        # recession), target = phi_k (0 matrix / 1 grain), tau in [0,dt]
        fi, fj = np.where((phi_k < 0.005) | (phi_k > 0.995))
        sd = rng.integers(0, fi.size, n_dn)
        X_dn = np.column_stack([xs[fi[sd]], xs[fj[sd]], rng.uniform(0, dt, n_dn)])
        phi_dn = phi_k[fi[sd], fj[sd]].reshape(-1, 1)
        return {"X_f": X_f, "X_ic": X_ic, "phi_ic": phi_ic, "X_dn": X_dn, "phi_dn": phi_dn}

    def predict_grid(m):
        Xt = np.column_stack([flat, np.full(len(flat), dt)])
        return m(tf.constant(Xt, dtype=pm.DTYPE)).numpy().reshape(N, N)

    outdir = outdir or os.path.join(ROOT, "outputs", "b2")

    def save(traj, mses, done):
        R_pinn = [float(grain_radius(np.clip(g, 0, 1), dx)) for g in traj]
        R_ref = [float(grain_radius(g, dx)) for g in ref_grids[:len(traj)]]
        m = {"benchmark": "b2_grain_shrinkage", "scheme": "time-marched+denoising", "quick": quick,
             "units": "grid (Dx=1.5e-5 m)", "K_intervals": K, "dt": dt, "done": done,
             "intervals_completed": len(mses), "mse_per_interval": mses,
             "mse_vs_reference_mean": (float(np.mean(mses)) if mses else None),
             "R_pinn": R_pinn, "R_ref": R_ref, "t_boundaries": bnd.tolist(),
             "radius_mae": (float(np.mean(np.abs(np.array(R_pinn) - np.array(R_ref)))) if mses else None),
             "tf": info["tf_version"], "gpus": info["gpus"]}
        io.save_json(m, os.path.join(outdir, "metrics.json"))
        io.checkpoint.save_weights(model, os.path.join(outdir, "weights"))
        return m

    traj, mses = time_march(model, p, b2_loss, make_batch, predict_grid, ref_grids, dt,
                            adam_steps=adam_steps, lbfgs_iters=lbfgs_iters, max_restarts=2,
                            on_interval=lambda k, tr, ms: save(tr, ms, False))  # crash/timeout-safe
    metrics = save(traj, mses, True)
    try:
        viz.plots.plot_solution_vs_reference(traj[-1], ref_grids[-1], xs, xs,
                                             os.path.join(outdir, "solution_tfinal.png"))
    except Exception as e:  # noqa: BLE001
        print("plot skipped:", e)
    print(f"\n=== B2 time-marched+denoising: mean MSE_vs_ref={metrics['mse_vs_reference_mean']:.3e}  "
          f"radius_MAE={metrics['radius_mae']:.3e}  R_pinn={[round(r, 1) for r in metrics['R_pinn']]} ===")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--outdir", default=None); a = ap.parse_args()
    run(quick=a.quick, outdir=a.outdir)
