"""Multi-phase reference solver — fast CPU checks (NumPy only, no GPU).

Rigorous fast checks; the precise N=2->curvature-law (3.6%) and triple-junction
->120deg validations run at N=128, too slow for CI here.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from reference.pf_solver import grain_radius  # noqa: E402
from reference.pf_solver_multiphase import (MPParams, init_circle_2phase,  # noqa: E402
                                            init_t_junction, simulate, stable_dt)


def test_sum_constraint_conserved():
    N = 64
    p = MPParams(mu=1.0, sigma=1.0, eta=0.06)
    phis0 = init_t_junction(N)
    _, _, sum_err = simulate(phis0, p, 1.0 / N, dt=stable_dt(p, 1.0 / N),
                             t_final=0.01, save_every=50)
    assert np.max(sum_err) < 1e-10, f"sum constraint drift {np.max(sum_err):.2e}"


def test_n2_grain_shrinks():
    N = 64
    dx = 1.0 / N
    p = MPParams(mu=1.0, sigma=1.0, eta=0.06)
    phis0 = init_circle_2phase(N, 1.0, 0.30, p.eta)
    _, frames, sum_err = simulate(phis0, p, dx, dt=stable_dt(p, dx),
                                  t_final=0.02, save_every=100)
    R = np.array([grain_radius(f[0], dx) for f in frames])
    assert R[-1] < R[0], "grain should shrink under curvature"
    assert np.max(sum_err) < 1e-10
