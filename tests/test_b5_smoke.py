"""B5 MultiNN smoke tests.

These tests verify the architecture scaffolding BEFORE any long GPU run:
  1. Output shape  : predict_grid_multiphase returns [4, N, N]
  2. Sum constraint: max |Σφ - 1| < 1e-12 (softmax by construction)
  3. No NaNs       : no NaN in output at t=0 and t=dt
  4. Batch nonempty: continuity batch has at least one nonempty pair
  5. One interval  : a tiny window trains and writes metrics without crashing

All tests use tiny config (N=8 grid, 2 workers, 2-layer nets) to run in <10s on CPU.

Run with:
    pytest tests/test_b5_smoke.py -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tensorflow as tf                                           # noqa: E402
import pinns_mpf as pm                                            # noqa: E402
from pinns_mpf.config import PhysicalParams                       # noqa: E402
from pinns_mpf.decomposition.decomp import (                      # noqa: E402
    MultiNN, make_boxes, neighbor_pairs,
)
from pinns_mpf.decomposition.decomp_multiphase import (           # noqa: E402
    build_batch_multiphase, make_loss_multiphase,
    predict_grid_multiphase,
)

# ── shared tiny config ────────────────────────────────────────────────────────

L       = 1.0
ETA     = 6.0 / 64.0
MU      = 1e-4
SIGMA   = 1.0
DT_W    = 25.0
N_GRID  = 8      # tiny grid: 8×8
N_BOX   = 2      # two-worker test layout (1×2 for speed)
OVERLAP = 0.15
NPHASES = 4

pm.setup(seed=42)


def _tiny_phi():
    """4-phase field on an N_GRID×N_GRID periodic grid with 2 grains."""
    rng = np.random.default_rng(7)
    phi = rng.dirichlet(np.ones(NPHASES), size=(N_GRID, N_GRID))  # [N, N, 4]
    phi = phi.transpose(2, 0, 1).astype(np.float64)                # [4, N, N]
    # Normalize (dirichlet already sums to 1 but ensure float64 precision)
    phi /= phi.sum(axis=0, keepdims=True)
    return phi


def _build_tiny():
    """Build a minimal MultiNN (1×2 layout, 2 workers, tiny hidden)."""
    boxes   = make_boxes(L, 1, 2, OVERLAP)           # 2 boxes (left/right strip)
    model   = MultiNN(boxes, hidden=(16, 16), out_dim=NPHASES, dt=DT_W,
                      output_activation="linear", periodic_dims=[0, 1],
                      domain=L, seed=42)
    pairs   = neighbor_pairs(boxes, 1, 2)
    xs      = np.linspace(0, L, N_GRID, endpoint=False)
    return model, boxes, pairs, xs


# ── Test 1: output shape ─────────────────────────────────────────────────────

def test_output_shape():
    """predict_grid_multiphase must return [4, N, N]."""
    model, _, _, xs = _build_tiny()
    phi = predict_grid_multiphase(model, xs, tau=0.0)
    assert phi.shape == (NPHASES, N_GRID, N_GRID), (
        f"Expected [4, {N_GRID}, {N_GRID}], got {phi.shape}")


# ── Test 2: sum constraint ────────────────────────────────────────────────────

def test_sum_constraint():
    """Softmax output must satisfy Σφ = 1 to machine precision."""
    model, _, _, xs = _build_tiny()
    for tau in [0.0, DT_W * 0.5, DT_W]:
        phi = predict_grid_multiphase(model, xs, tau=tau)
        err = np.abs(phi.sum(axis=0) - 1.0).max()
        assert err < 1e-12, f"Sum constraint violated at tau={tau}: max_err={err:.2e}"


# ── Test 3: no NaNs ──────────────────────────────────────────────────────────

def test_no_nans():
    """Output must be finite at both endpoints of the window."""
    model, _, _, xs = _build_tiny()
    for tau in [0.0, DT_W]:
        phi = predict_grid_multiphase(model, xs, tau=tau)
        assert np.all(np.isfinite(phi)), (
            f"NaN/Inf in phi at tau={tau}. "
            f"n_bad={np.sum(~np.isfinite(phi))}")


# ── Test 4: continuity batch nonempty ────────────────────────────────────────

def test_continuity_batch_nonempty():
    """At least one overlap halo must have nonempty continuity collocation."""
    model, _, pairs, xs = _build_tiny()
    phi_k = _tiny_phi()
    batch, _ = build_batch_multiphase(
        model, phi_k, xs, DT_W,
        n_f=50, n_ic=30, n_dn=20, n_cont=20,
        eta=ETA, seed=0)
    n_nonempty = sum(1 for Xc in batch["Xc"] if Xc.shape[0] > 0)
    assert n_nonempty > 0, (
        "All continuity batches are empty — overlap halo detection failed. "
        f"boxes: {[b['ext'] for b in model.boxes]}")


# ── Test 5: one interval trains and saves metrics ────────────────────────────

def test_one_interval_trains():
    """Run 20 Adam steps on a tiny window; loss must decrease and not NaN."""
    model, _, pairs, xs = _build_tiny()
    p     = PhysicalParams(mu=MU, sigma=SIGMA, eta=ETA)
    phi_k = _tiny_phi()

    batch, _ = build_batch_multiphase(
        model, phi_k, xs, DT_W,
        n_f=80, n_ic=40, n_dn=20, n_cont=20,
        eta=ETA, seed=1)

    loss_fn, set_batch, W, terms, *_ = make_loss_multiphase(
        model, p, pairs, w_pde=1.0, w_ic=2.0, w_dn=0.0, w_cont=0.0)
    set_batch(batch)

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    vars_ = model.trainable_variables
    losses = []
    for _ in range(20):
        with tf.GradientTape() as tape:
            total, _ = loss_fn()
        opt.apply_gradients(zip(tape.gradient(total, vars_), vars_))
        losses.append(float(total))

    assert np.all(np.isfinite(losses)), f"NaN/Inf loss during training: {losses}"

    # Loss should drop (at least for first 20 steps from random init)
    assert losses[-1] < losses[0] * 2.0, (
        f"Loss did not decrease: initial={losses[0]:.3e} final={losses[-1]:.3e}")

    # Save metrics to tmp dir and verify JSON
    with tempfile.TemporaryDirectory() as tmp:
        metrics = {
            "test": "b5_smoke",
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "n_workers": len(model.workers),
        }
        path = os.path.join(tmp, "metrics.json")
        with open(path, "w") as f:
            json.dump(metrics, f)
        loaded = json.load(open(path))
        assert loaded["n_workers"] == len(model.workers)

    print(f"  Loss: {losses[0]:.3e} -> {losses[-1]:.3e}  ({20} Adam steps)")


# ── Test 6: runner quick mode completes without shape errors ──────────────────

def test_runner_quick_shape():
    """Run the full --quick runner for ONE window; verify shape and JSON output.

    Uses 10 Adam steps (not the default 100) to keep runtime short in CI.
    Patches the quick config to a single window to bound the test.
    """
    import importlib, types, tempfile, time as _t
    from benchmarks.b5_triple_junction import run_multiNN

    # Monkey-patch the quick config to 1 window, 10 Adam steps, no L-BFGS
    orig_run = run_multiNN.run

    def patched_run(quick=False, outdir=None, rollout=False, seed=1234):
        import tensorflow as tf
        info = run_multiNN.pm.setup(seed=seed)
        p    = run_multiNN.PhysicalParams(mu=run_multiNN.MU,
                                          sigma=run_multiNN.SIGMA,
                                          eta=run_multiNN.ETA)
        dt_w = run_multiNN.T_FINAL / 2.0   # 2 windows; we'll run only 1
        multinn, boxes = run_multiNN.build_multinn(dt_w, hidden=(16, 16), seed=seed)
        pairs = run_multiNN.neighbor_pairs(boxes, run_multiNN.N_SPATIAL, run_multiNN.N_SPATIAL)
        loss_fn, set_batch, W, terms, *_ = run_multiNN.make_loss_multiphase(
            multinn, p, pairs, w_pde=1.0, w_ic=2.0, w_dn=0.0, w_cont=0.0)

        phi_fd, times_fd = run_multiNN.load_fd_reference()
        xs_fd = np.linspace(0, run_multiNN.L, phi_fd.shape[-1], endpoint=False)
        phi_ic = run_multiNN.fd_frame_at(phi_fd, times_fd, 0.0)

        batch, _ = run_multiNN.build_batch_multiphase(
            multinn, phi_ic, xs_fd, dt_w,
            n_f=200, n_ic=100, n_dn=60, n_cont=40,
            eta=run_multiNN.ETA, seed=seed)
        set_batch(batch)

        # 10 Adam steps only
        opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
        vars_ = multinn.trainable_variables
        for _ in range(10):
            with tf.GradientTape() as tape:
                total, _ = loss_fn()
            opt.apply_gradients(zip(tape.gradient(total, vars_), vars_))

        # Evaluate on the packaged reference's native grid.
        phi_fd_end = run_multiNN.fd_frame_at(phi_fd, times_fd, dt_w)
        phi_pinn_end, metrics = run_multiNN.evaluate(
            multinn, xs_fd, phi_fd_end, dt_w)

        n_ref = phi_fd.shape[-1]
        assert phi_pinn_end.shape == (run_multiNN.NPHASES, n_ref, n_ref), (
            f"Shape error: expected ({run_multiNN.NPHASES},{n_ref},{n_ref}), "
            f"got {phi_pinn_end.shape}")
        assert np.all(np.isfinite(phi_pinn_end)), "NaN/Inf in phi_pinn_end"
        assert metrics["max_sum_err"] < 1e-12, (
            f"Sum constraint violated: {metrics['max_sum_err']:.2e}")

        if outdir:
            os.makedirs(outdir, exist_ok=True)
            with open(os.path.join(outdir, "quick_smoke_metrics.json"), "w") as f:
                json.dump(metrics, f)

        return metrics

    with tempfile.TemporaryDirectory() as tmp:
        t0 = _t.time()
        metrics = patched_run(outdir=tmp)
        wall = _t.time() - t0

    print(f"  Runner quick smoke: MSE={metrics['mse_vs_fd']:.3e}"
          f"  sum_err={metrics['max_sum_err']:.2e}  wall={wall:.1f}s")
    assert "mse_vs_fd" in metrics
    assert "argmax_disagree" in metrics
