"""2D single-phase dimensionless MPF physics (paper Eq. 17, 2D Laplacian).

    phi_t = mu [ sigma ( phi_xx + phi_yy + (pi^2/2 eta^2)(2 phi - 1) )
                 + (pi/eta) sqrt(phi(1-phi)) * delta_g ]

Same equation as the FD reference solver (reference.pf_solver) and the 1D case,
extended to two spatial dimensions. Derivatives via nested GradientTapes; float64,
GPU-resident.
"""
from __future__ import annotations

import tensorflow as tf

from ..config import DTYPE, PhysicalParams
from .single_phase import PI, dh_dphi, double_well


def profile_residual_2d(model, x, y, t, p: PhysicalParams):
    """Equilibrium diffuse-interface profile residual (UNSUPERVISED): |grad phi|^2 - (pi^2/eta^2)phi(1-phi).

    The stationary 1D Allen-Cahn profile of width eta satisfies |grad phi|^2 = (pi^2/eta^2) phi(1-phi)
    (first integral of phi'' + (pi^2/2eta^2)(2phi-1)=0). This is a property of eta and the PDE -- NOT
    the FD solution or radius law -- so penalizing it stays unsupervised. It sharpens the interface to
    the correct eta-width, reducing over-diffuse halos (which inflate |grad phi|^2 above the relation)
    and, since a correct-width interface carries the right curvature velocity, helps the late mobility.
    Exact for a flat interface; for R>>eta the curvature correction is O(eta/R), small in the band."""
    eta = tf.constant(p.eta, dtype=DTYPE)
    with tf.GradientTape(persistent=True) as tp:
        tp.watch(x)
        tp.watch(y)
        phi = model(tf.concat([x, y, t], axis=1))
    phi_x = tp.gradient(phi, x)
    phi_y = tp.gradient(phi, y)
    del tp
    return (phi_x ** 2 + phi_y ** 2) - (PI ** 2 / eta ** 2) * phi * (1.0 - phi)


def pde_residual_2d(model, x, y, t, p: PhysicalParams, project=False, eps=1e-3):
    """Residual f = phi_t - RHS at collocation columns x, y, t (each [N,1]).

    ``project`` switches the target to the bound-aware EFFECTIVE RHS that matches the
    clipped explicit reference integrator (reference.pf_solver.projected_dphi_dt):
    in the saturated bulk where the raw RHS would push phi out of [0,1], the effective
    rate is zero. This removes the artificial bulk-grow residual that previously had to
    be fought with heavy denoising. The bound test uses stop-gradient on phi (the mask
    is a selector, not a differentiable threshold): where projected, no PDE gradient
    flows; elsewhere the full residual gradient flows as usual."""
    eta = tf.constant(p.eta, dtype=DTYPE)
    mu = tf.constant(p.mu, dtype=DTYPE)
    sigma = tf.constant(p.sigma, dtype=DTYPE)
    dg = tf.constant(p.delta_g, dtype=DTYPE)

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x)
        tape2.watch(y)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x)
            tape1.watch(y)
            tape1.watch(t)
            phi = model(tf.concat([x, y, t], axis=1))
        phi_x = tape1.gradient(phi, x)
        phi_y = tape1.gradient(phi, y)
        phi_t = tape1.gradient(phi, t)
        del tape1
    phi_xx = tape2.gradient(phi_x, x)
    phi_yy = tape2.gradient(phi_y, y)
    del tape2

    lap = phi_xx + phi_yy
    rhs = mu * (sigma * (lap + double_well(phi, eta)) + dh_dphi(phi, eta) * dg)
    if project:
        phi_c = tf.stop_gradient(phi)
        rhs_c = tf.stop_gradient(rhs)
        out_low = (phi_c <= eps) & (rhs_c < 0.0)
        out_high = (phi_c >= 1.0 - eps) & (rhs_c > 0.0)
        rhs = tf.where(out_low | out_high, tf.zeros_like(rhs), rhs)
    return phi_t - rhs
