"""Single-phase dimensionless MPF physics (paper Eqs. 17-18, 24).

PDE (1D space + time), constant driving force, isotropic mobility/energy:

    phi_t = mu * [ sigma * (phi_xx + (pi^2 / (2 eta^2)) (2 phi - 1))
                   + (pi / eta) sqrt(phi (1 - phi)) * delta_g ]

The double-well/gradient combination vanishes for the equilibrium sin-profile, so
the traveling-wave speed is exactly ``v_n = mu * delta_g`` (see reference.analytic).

Derivatives are taken with autodiff (nested GradientTapes) — GPU-resident, float64.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from ..config import DTYPE, PhysicalParams

PI = tf.constant(np.pi, dtype=DTYPE)


# Tiny floor inside the sqrt: dh/dphi = (pi/eta) sqrt(phi(1-phi)) has a 1/sqrt
# derivative that diverges as phi -> 0/1 (the saturated bulk). The floor keeps the
# loss gradient finite there (physically negligible: sqrt floor ~ 1e-6) so L-BFGS
# stays well conditioned.
_SQRT_EPS = tf.constant(1e-12, dtype=DTYPE)


def dh_dphi(phi, eta):
    """d h / d phi = (pi / eta) sqrt(phi (1 - phi))  (Eq. 18)."""
    return (PI / eta) * tf.sqrt(tf.maximum(phi * (1.0 - phi), 0.0) + _SQRT_EPS)


def double_well(phi, eta):
    """(pi^2 / (2 eta^2)) (2 phi - 1) — the local part of the gradient term."""
    return (PI ** 2 / (2.0 * eta ** 2)) * (2.0 * phi - 1.0)


def pde_residual(model, x, t, p: PhysicalParams):
    """Residual f = phi_t - RHS, evaluated at collocation columns x, t (shape [N,1])."""
    eta = tf.constant(p.eta, dtype=DTYPE)
    mu = tf.constant(p.mu, dtype=DTYPE)
    sigma = tf.constant(p.sigma, dtype=DTYPE)
    dg = tf.constant(p.delta_g, dtype=DTYPE)

    with tf.GradientTape(persistent=True) as t2:
        t2.watch(x)
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(x)
            t1.watch(t)
            phi = model(tf.concat([x, t], axis=1))
        phi_x = t1.gradient(phi, x)
        phi_t = t1.gradient(phi, t)
        del t1
    phi_xx = t2.gradient(phi_x, x)
    del t2

    rhs = mu * (sigma * (phi_xx + double_well(phi, eta)) + dh_dphi(phi, eta) * dg)
    return phi_t - rhs
