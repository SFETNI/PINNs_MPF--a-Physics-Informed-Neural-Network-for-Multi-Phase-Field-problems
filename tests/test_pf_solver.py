"""Reference FD phase-field solver vs the analytic curvature law (CPU/NumPy only).

No TensorFlow / no GPU — safe to run anytime. Validates that a circular grain with
no driving force shrinks as R(t)^2 = R0^2 - 2 mu sigma t.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from reference.pf_solver import (PFParams, curvature_law_radius, init_circle,  # noqa: E402
                                 simulate, stable_dt)


def test_curvature_law():
    N, L, R0 = 128, 1.0, 0.30
    dx = L / N
    p = PFParams(mu=1.0, sigma=1.0, eta=7 * dx, delta_g=0.0)
    phi0, _ = init_circle(N, L, R0, p.eta)
    t, _, R = simulate(phi0, p, dx, dt=stable_dt(p, dx), t_final=0.028, save_every=200)

    m = R > 0.15  # well-resolved portion (diffuse interface biases small radii)
    slope = np.polyfit(t[m], R[m] ** 2, 1)[0]
    # theory is -2*mu*sigma = -2; finite eta gives a few-% correction
    assert abs(slope - (-2.0)) / 2.0 < 0.05, f"d(R^2)/dt={slope:.4f} (expected ~-2)"
    assert np.all(np.diff(R) <= 1e-9), "radius must be (weakly) monotonically decreasing"
    Rth = curvature_law_radius(t, R0, p.mu, p.sigma)
    assert np.max(np.abs(R[m] - Rth[m])) < 0.01, "radius deviates from curvature law"
