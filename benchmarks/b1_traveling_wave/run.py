"""Benchmark 1 — traveling wave interface (single NN), GPU-native float64.

Re-validates the modernized PINNs-MPF core against the exact analytic solution
(paper Eq. 26). This is the template for Benchmarks 2-5.

    python benchmarks/b1_traveling_wave/run.py            # full run
    python benchmarks/b1_traveling_wave/run.py --quick    # fast (used by the test)
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
from pinns_mpf.sampling import latin_hypercube  # noqa: E402
from reference.analytic import traveling_wave, traveling_wave_grid  # noqa: E402


def build_dataset(p, n_f, n_ic, n_b, x_range, t_range, seed,
                  frac_band=0.6, band_halfwidth=0.9):
    """Training data. Collocation concentrates a fraction of points in the moving
    interface band x ~ v_n*t +/- band_halfwidth*eta (the paper's interface-focused
    meshing) and scatters the rest uniformly (LHS) over the domain. The band uses
    only physical inputs (v_n, eta, initial interface at x=0), not the solution."""
    xmin, xmax = x_range
    tmin, tmax = t_range
    lb, ub = [xmin, tmin], [xmax, tmax]
    rng = np.random.default_rng(seed)

    x_ic = np.linspace(xmin, xmax, n_ic)
    X_ic = np.column_stack([x_ic, np.full_like(x_ic, tmin)])
    phi_ic = traveling_wave(x_ic, tmin, p.eta, p.v_n).reshape(-1, 1)

    tb = np.linspace(tmin, tmax, n_b)
    X_lb = np.column_stack([np.full_like(tb, xmin), tb])  # left edge -> phi = 1
    X_ub = np.column_stack([np.full_like(tb, xmax), tb])  # right edge -> phi = 0

    n_band = int(frac_band * n_f)
    t_band = rng.uniform(tmin, tmax, n_band)
    x_band = np.clip(p.v_n * t_band + rng.uniform(-band_halfwidth * p.eta,
                                                  band_halfwidth * p.eta, n_band), xmin, xmax)
    X_band = np.column_stack([x_band, t_band])
    X_uniform = latin_hypercube(lb, ub, n_f - n_band, seed=seed)
    X_f = np.vstack([X_band, X_uniform])

    return {"X_f": X_f, "X_ic": X_ic, "phi_ic": phi_ic, "X_lb": X_lb, "X_ub": X_ub}


def _interface_position(phi_col, x):
    """x where phi crosses 0.5 (linear interp); nan if no crossing."""
    f = phi_col - 0.5
    s = np.where(np.diff(np.sign(f)) != 0)[0]
    if s.size == 0:
        return np.nan
    i = s[0]
    return x[i] - f[i] * (x[i + 1] - x[i]) / (f[i + 1] - f[i])


def fit_interface_velocity(phi_pred, x, t):
    pos = np.array([_interface_position(phi_pred[:, j], x) for j in range(len(t))])
    m = ~np.isnan(pos)
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(t[m], pos[m], 1)[0])


def run(quick=False, outdir=None, make_plots=True, seed=1234,
        x_range=(-1.0, 1.0), t_range=(0.0, 1.0), lr=1e-3,
        layers=(2, 32, 32, 32, 32, 1)):
    if quick:
        # fast config for the locked regression test (~1-2 min on the A10)
        n_f, n_ic, n_b, adam_steps, lbfgs_iters, max_restarts, test_n = 1500, 80, 80, 400, 250, 1, 150
    else:
        # high-precision config (MSE ~1e-6..1e-7; ~30-50 min on the A10 in float64)
        n_f, n_ic, n_b, adam_steps, lbfgs_iters, max_restarts, test_n = 6000, 120, 120, 1500, 1500, 5, 400

    info = pm.setup(seed=seed)
    print("env:", info)
    p = pm.PhysicalParams()  # eta=1, mu=1e-6, sigma=1, delta_g=5e5 -> v_n=0.5
    print(f"physical params: eta={p.eta}, mu={p.mu}, sigma={p.sigma}, "
          f"delta_g={p.delta_g}, v_n={p.v_n}")

    batch = build_dataset(p, n_f, n_ic, n_b, x_range, t_range, seed)
    model = pm.MLP(list(layers), lb=[x_range[0], t_range[0]], ub=[x_range[1], t_range[1]],
                   output_activation="sigmoid", seed=seed)
    print(f"model: layers={list(layers)}, params={model.num_parameters}")

    trainer = pm.SinglePhaseTrainer(model, p, batch, lr=lr)
    trainer.fit(adam_steps=adam_steps, lbfgs_iters=lbfgs_iters, max_restarts=max_restarts)

    X_test, phi_true, x, t = traveling_wave_grid(p.eta, p.v_n, nx=test_n, nt=test_n,
                                                 x_range=x_range, t_range=t_range)
    phi_pred = trainer.evaluate(X_test).reshape(test_n, test_n, order="F")

    mse = float(np.mean((phi_pred - phi_true) ** 2))
    mae = float(np.mean(np.abs(phi_pred - phi_true)))
    l2rel = float(np.linalg.norm(phi_pred - phi_true) / np.linalg.norm(phi_true))
    v_fit = fit_interface_velocity(phi_pred, x, t)
    metrics = {
        "benchmark": "b1_traveling_wave", "quick": quick,
        "mse": mse, "mae": mae, "l2_relative": l2rel,
        "v_n_true": p.v_n, "v_n_fit": v_fit, "v_n_rel_err": abs(v_fit - p.v_n) / p.v_n,
        "final_loss": trainer.history[-1], "tf": info["tf_version"], "gpus": info["gpus"],
    }
    print("\n=== Benchmark 1 metrics ===")
    for k in ("mse", "mae", "l2_relative", "v_n_true", "v_n_fit", "v_n_rel_err"):
        print(f"  {k:14s}: {metrics[k]:.6e}" if isinstance(metrics[k], float) else f"  {k}: {metrics[k]}")

    outdir = outdir or os.path.join(ROOT, "outputs", "b1")
    io.save_json(metrics, os.path.join(outdir, "metrics.json"))
    io.checkpoint.save_weights(model, os.path.join(outdir, "weights"))
    if make_plots:
        viz.plots.plot_solution_vs_reference(phi_pred, phi_true, x, t,
                                             os.path.join(outdir, "solution.png"))
        viz.plots.plot_loss(trainer.history, os.path.join(outdir, "loss.png"))
    print(f"saved -> {outdir}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    run(quick=a.quick, make_plots=not a.no_plots, outdir=a.outdir)
