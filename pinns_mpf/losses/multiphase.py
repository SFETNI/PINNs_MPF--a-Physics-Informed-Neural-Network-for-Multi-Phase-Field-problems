"""Loss terms for the multi-phase benchmark (B5 triple junction).

The network outputs N logits; softmax -> phase fields (sum=1 by construction). IC
and BC use reference (FD) data as targets (a data-assisted, well-posed setup; the
legacy framework likewise propagated boundary data between subdomains). PDE term is
the multi-phase residual.

batch keys: X_f [N,3], X_ic [N,3], phi_ic [N,P], X_bc [N,3], phi_bc [N,P].
"""
from __future__ import annotations

import tensorflow as tf

from ..config import PhysicalParams
from ..physics.multiphase import loss_pde_multiphase, phases_from_logits


def loss_ic(model, X_ic, phi_ic):
    return tf.reduce_mean(tf.square(phases_from_logits(model(X_ic)) - phi_ic))


def loss_bc_data(model, X_bc, phi_bc):
    return tf.reduce_mean(tf.square(phases_from_logits(model(X_bc)) - phi_bc))


def make_total_loss_multiphase(num_phases: int):
    """Return a total_loss(model, batch, p, weights) closure bound to num_phases
    (matches the SinglePhaseTrainer loss_fn signature)."""
    def total_loss_multiphase(model, batch, p: PhysicalParams, weights=(1.0, 1.0, 1.0)):
        w_pde, w_ic, w_bc = weights
        l_pde = loss_pde_multiphase(model, batch["X_f"], p, num_phases)
        l_ic = loss_ic(model, batch["X_ic"], batch["phi_ic"])
        l_bc = loss_bc_data(model, batch["X_bc"], batch["phi_bc"])
        total = w_pde * l_pde + w_ic * l_ic + w_bc * l_bc
        return total, {"total": total, "pde": l_pde, "ic": l_ic, "bc": l_bc}
    return total_loss_multiphase
