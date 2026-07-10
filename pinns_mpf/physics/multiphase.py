"""Multi-phase (N order parameters) dimensionless MPF residual — for B5 triple junction.

Same model as the reference solver (reference.pf_solver_multiphase, Eqs. 22-23):

    I_alpha       = phi_xx + phi_yy + (pi^2/eta^2) phi_alpha
    dphi_alpha/dt = mu sigma (I_alpha - mean_beta I_beta)

Design improvement over the legacy N-network scheme: a *single* network outputs N
logits and a softmax maps them to phase fields, so phi_alpha in (0,1) and
sum_alpha phi_alpha = 1 hold *by construction* — the sum-constraint loss the legacy
needed becomes unnecessary.

Derivatives are computed via scalar gradient calls (not batch_jacobian/pfor).
Key implementation rule: all tensor slices used as gradient targets must be created
INSIDE the relevant GradientTape context — slices computed after tape.__exit__ return
None from tape.gradient even when the parent tensor was inside the tape. float64, GPU.

Second-order Laplacian: only phi_xx and phi_yy are needed (diagonal Hessian).
We use the reduce_sum identity: tape.gradient(y_vec, x) where y_vec is a vector
equals d(sum(y_vec))/dx, which for independent samples gives the per-sample Jacobian.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from ..config import DTYPE, PhysicalParams


def phases_from_logits(logits):
    """Softmax over the phase axis -> phi in (0,1), sum_alpha phi = 1."""
    return tf.nn.softmax(logits, axis=1)


def pde_residual_multiphase(model, X, p: PhysicalParams, num_phases: int):
    """Residual f_alpha = phi_t,alpha - mu sigma (I_alpha - mean_beta I_beta).

    X: collocation points [N, 3] = (x, y, t). Returns [N, num_phases].

    Uses scalar gradients (no batch_jacobian/pfor) to avoid pfor JIT compilation
    overhead, which is prohibitive for deep nets (64×6) on GPU in eager mode.
    Verified equivalent to batch_jacobian to machine precision (< 1e-17).
    """
    prefactor = tf.constant(np.pi ** 2 / p.eta ** 2, dtype=DTYPE)
    mu_c     = tf.constant(p.mu,    dtype=DTYPE)
    sigma_c  = tf.constant(p.sigma, dtype=DTYPE)
    P = num_phases

    with tf.GradientTape(persistent=True) as t2:
        t2.watch(X)
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(X)
            phi = phases_from_logits(model(X))         # [N, P]
            # Slices must be created INSIDE the tape so t1/t2 track them.
            phi_phases = [phi[:, a] for a in range(P)]
        # First-order grads per phase; computed inside t2 so t2 can second-diff them.
        grads1_x, grads1_y, phi_t_list = [], [], []
        for a in range(P):
            g = t1.gradient(phi_phases[a], X)          # [N, 3]: dphi_a/d(x,y,t)
            grads1_x.append(g[:, 0])                   # phi_x_a — inside t2 context
            grads1_y.append(g[:, 1])                   # phi_y_a — inside t2 context
            phi_t_list.append(g[:, 2])
        del t1

    # Second derivatives (Laplacian diagonal only).
    # t2.gradient(vec, X) = d(sum(vec))/dX; for independent samples this equals
    # the per-sample second derivative (only diagonal entries are non-zero).
    phi_t = tf.stack(phi_t_list, axis=1)               # [N, P]
    lap_list = []
    for a in range(P):
        phi_xx = t2.gradient(grads1_x[a], X)[:, 0]    # [N]
        phi_yy = t2.gradient(grads1_y[a], X)[:, 1]    # [N]
        lap_list.append(phi_xx + phi_yy)
    del t2

    lap   = tf.stack(lap_list, axis=1)                 # [N, P]
    I     = lap + prefactor * phi                      # [N, P]
    I_mean = tf.reduce_mean(I, axis=1, keepdims=True)  # [N, 1]
    return phi_t - mu_c * sigma_c * (I - I_mean)      # [N, P]


def loss_pde_multiphase(model, X_f, p: PhysicalParams, num_phases: int):
    r = pde_residual_multiphase(model, X_f, p, num_phases)
    return tf.reduce_mean(tf.square(r))
