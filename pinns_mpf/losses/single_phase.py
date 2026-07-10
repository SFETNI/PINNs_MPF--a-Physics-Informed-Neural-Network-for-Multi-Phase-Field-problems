"""Loss terms for the single-phase benchmark (PDE / IC / Dirichlet BC).

Equal weighting matches the legacy Benchmark-1 setup; weights are exposed so they
can be tuned without touching the math.
"""
from __future__ import annotations

import tensorflow as tf

from ..config import DTYPE, PhysicalParams
from ..physics.single_phase import pde_residual


def loss_ic(model, X_ic, phi_ic):
    return tf.reduce_mean(tf.square(phi_ic - model(X_ic)))


def loss_bc_dirichlet(model, X_lb, X_ub, phi_lb=1.0, phi_ub=0.0):
    """phi = phi_lb on the left edge, phi = phi_ub on the right edge."""
    lb = tf.constant(phi_lb, dtype=DTYPE)
    ub = tf.constant(phi_ub, dtype=DTYPE)
    return (tf.reduce_mean(tf.square(model(X_lb) - lb))
            + tf.reduce_mean(tf.square(model(X_ub) - ub)))


def loss_pde(model, X_f, p: PhysicalParams):
    r = pde_residual(model, X_f[:, 0:1], X_f[:, 1:2], p)
    return tf.reduce_mean(tf.square(r))


def total_loss(model, batch, p: PhysicalParams, weights=(1.0, 1.0, 1.0)):
    """batch = dict(X_f, X_ic, phi_ic, X_lb, X_ub). Returns (total, components dict)."""
    w_pde, w_ic, w_bc = weights
    l_pde = loss_pde(model, batch["X_f"], p)
    l_ic = loss_ic(model, batch["X_ic"], batch["phi_ic"])
    l_bc = loss_bc_dirichlet(model, batch["X_lb"], batch["X_ub"])
    total = w_pde * l_pde + w_ic * l_ic + w_bc * l_bc
    return total, {"total": total, "pde": l_pde, "ic": l_ic, "bc": l_bc}
