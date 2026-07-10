"""Training orchestration: Adam warm-up + GPU-native L-BFGS polish.

The legacy "wheel of optimizers" alternated Adam and scipy L-BFGS while pinned to
CPU. Here the single-NN benchmark uses Adam to escape the initial transient, then
TFP L-BFGS (on GPU, float64) to drive the loss down for high precision. Everything
stays on the GPU; the time-windowed / multi-NN orchestration (Benchmarks 2-5) will
extend this in later development.
"""
from __future__ import annotations

import time

import numpy as np
import tensorflow as tf

from .config import DTYPE, PhysicalParams
from .losses.single_phase import total_loss
from .optimizers import lbfgs


def _to_const(batch: dict) -> dict:
    return {k: tf.constant(np.asarray(v), dtype=DTYPE) for k, v in batch.items()}


class SinglePhaseTrainer:
    def __init__(self, model, params: PhysicalParams, batch: dict,
                 weights=(1.0, 1.0, 1.0), lr: float = 1e-3, loss_fn=total_loss):
        self.model = model
        self.p = params
        self.batch = _to_const(batch)
        self.weights = weights
        self.loss_fn = loss_fn  # 1D total_loss (default) or total_loss_2d for B2+
        self.opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.history: list[dict] = []

    def _scalar_loss(self):
        total, _ = self.loss_fn(self.model, self.batch, self.p, self.weights)
        return total

    @tf.function
    def _adam_step(self):
        with tf.GradientTape() as tape:
            total, comps = self.loss_fn(self.model, self.batch, self.p, self.weights)
        grads = tape.gradient(total, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return comps

    def adam(self, steps: int, log_every: int = 500):
        t0 = time.time()
        comps = None
        for k in range(steps):
            comps = self._adam_step()
            if log_every and (k % log_every == 0 or k == steps - 1):
                c = {kk: float(vv) for kk, vv in comps.items()}
                c["phase"], c["step"] = "adam", k
                self.history.append(c)
                parts = " ".join(f"{kk}={c[kk]:.3e}" for kk in comps)  # generic (bc may be absent)
                print(f"[adam {k:5d}] {parts}")
        print(f"  adam: {steps} steps in {time.time() - t0:.1f}s")
        return comps

    def lbfgs(self, max_iterations: int = 5000, **kw):
        t0 = time.time()
        res = lbfgs.minimize(self.model, self._scalar_loss,
                             max_iterations=max_iterations, **kw)
        _, comps = self.loss_fn(self.model, self.batch, self.p, self.weights)
        c = {kk: float(vv) for kk, vv in comps.items()}
        c["phase"], c["lbfgs_iters"] = "lbfgs", int(res.num_iterations)
        self.history.append(c)
        print(f"  lbfgs: {int(res.num_iterations)} iters, converged={bool(res.converged)}, "
              f"failed={bool(res.failed)}, total={c['total']:.3e} in {time.time() - t0:.1f}s")
        return res

    def fit(self, adam_steps: int = 2000, lbfgs_iters: int = 5000, max_restarts: int = 5):
        self.adam(adam_steps)
        self.lbfgs(lbfgs_iters, max_restarts=max_restarts)
        return self.history

    def evaluate(self, X) -> np.ndarray:
        return self.model(tf.constant(np.asarray(X), dtype=DTYPE)).numpy()


def time_march(model, params, loss_fn, make_batch, predict_grid, ref_grids, dt,
               adam_steps=300, lbfgs_iters=200, max_restarts=1, lr=1e-3, log=True,
               on_interval=None):
    """Discrete-in-time PINNs-MPF scheme (the original framework's core).

    Solve [0,T] as a sequence of short intervals [t_k, t_k+dt]. Each interval is a
    well-posed short-time IVP: IC at local time tau=0 is the previous interval's
    prediction (moving IC), the network uses LOCAL time tau in [0, dt] (so its
    t-feature spans the full range and avoids the long-horizon causality failure).
    Weights persist across intervals = transfer of learning. Per-interval predictions
    assemble the full trajectory.

    Callbacks (benchmark-specific):
      make_batch(phi_k, k) -> dict batch (IC sampled from grid phi_k at tau=0 + collocation in [0,dt])
      predict_grid(model)  -> phi grid at tau=dt (single phase [N,N] or multi [P,N,N])
    ref_grids: reference phi grids at interval boundaries t_0..t_K (len = n_intervals+1).
    Returns (trajectory[list of grids t_0..t_K], per_interval_mse[list]).
    """
    n_intervals = len(ref_grids) - 1
    phi_k = np.asarray(ref_grids[0])
    traj, mses = [phi_k], []
    for k in range(n_intervals):
        batch = make_batch(phi_k, k)
        tr = SinglePhaseTrainer(model, params, batch, lr=lr, loss_fn=loss_fn)
        tr.adam(adam_steps, log_every=0)
        tr.lbfgs(lbfgs_iters, max_restarts=max_restarts)
        phi_next = np.asarray(predict_grid(model))
        mse = float(np.mean((phi_next - np.asarray(ref_grids[k + 1])) ** 2))
        traj.append(phi_next); mses.append(mse)
        if log:
            print(f"  [interval {k + 1}/{n_intervals}] t={(k + 1) * dt:.2f} "
                  f"mse_vs_ref={mse:.3e}")
        if on_interval is not None:        # per-interval checkpoint (crash/timeout-safe)
            on_interval(k, traj, mses)
        phi_k = phi_next
    return traj, mses
