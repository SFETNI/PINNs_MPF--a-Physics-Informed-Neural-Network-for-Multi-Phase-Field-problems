"""Analytic reference solutions (ground truth for validation).

Benchmark 1 — traveling wave interface (paper Eq. 26):

    phi(x, t) = 1                                  for x < v_n t - eta/2
              = 1/2 - 1/2 sin(pi/eta (x - v_n t))  for |x - v_n t| <= eta/2
              = 0                                  for x > v_n t + eta/2

with interface speed v_n = mu * delta_g. This is an exact solution of the
single-phase MPF PDE in pinns_mpf.physics.single_phase.
"""
from __future__ import annotations

import numpy as np


def traveling_wave(x, t, eta: float, v_n: float):
    """Vectorized analytic phi(x, t). x, t broadcast together."""
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    xi = x - v_n * t
    return np.where(
        xi < -eta / 2.0, 1.0,
        np.where(xi > eta / 2.0, 0.0, 0.5 - 0.5 * np.sin(np.pi / eta * xi)),
    )


def traveling_wave_grid(eta, v_n, nx=1000, nt=1000, x_range=(-1.0, 1.0), t_range=(0.0, 1.0)):
    """Return (X_test [nx*nt, 2], phi_true [nx, nt], x [nx], t [nt]).

    phi_true[i, j] = phi(x[i], t[j]); X_test rows are (x, t) in column-major (F)
    order so ``phi_pred.reshape(nx, nt, order='F')`` aligns with phi_true.
    """
    x = np.linspace(x_range[0], x_range[1], nx)
    t = np.linspace(t_range[0], t_range[1], nt)
    Xg, Tg = np.meshgrid(x, t, indexing="ij")  # [nx, nt]
    phi_true = traveling_wave(Xg, Tg, eta, v_n)
    X_test = np.column_stack([Xg.flatten(order="F"), Tg.flatten(order="F")])
    return X_test, phi_true, x, t
