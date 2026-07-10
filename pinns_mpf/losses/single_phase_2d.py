"""Loss terms for the 2D single-phase benchmark (B2: grain shrinkage).

batch keys: X_f [N,3]=(x,y,t), X_ic [N,3], phi_ic [N,1], X_bc [N,3] (outer edges).
BC is Dirichlet phi=0 on the domain boundary (matrix phase surrounds the grain,
which stays interior).
"""
from __future__ import annotations

import tensorflow as tf

from ..config import PhysicalParams
from ..physics.single_phase_2d import pde_residual_2d


def loss_ic(model, X_ic, phi_ic):
    return tf.reduce_mean(tf.square(phi_ic - model(X_ic)))


def loss_bc_zero(model, X_bc):
    return tf.reduce_mean(tf.square(model(X_bc)))


def loss_pde(model, X_f, p: PhysicalParams):
    r = pde_residual_2d(model, X_f[:, 0:1], X_f[:, 1:2], X_f[:, 2:3], p)
    return tf.reduce_mean(tf.square(r))


def total_loss_2d(model, batch, p: PhysicalParams, weights=(1.0, 1.0, 1.0)):
    w_pde, w_ic, w_bc = weights
    l_pde = loss_pde(model, batch["X_f"], p)
    l_ic = loss_ic(model, batch["X_ic"], batch["phi_ic"])
    l_bc = loss_bc_zero(model, batch["X_bc"])
    total = w_pde * l_pde + w_ic * l_ic + w_bc * l_bc
    return total, {"total": total, "pde": l_pde, "ic": l_ic, "bc": l_bc}
