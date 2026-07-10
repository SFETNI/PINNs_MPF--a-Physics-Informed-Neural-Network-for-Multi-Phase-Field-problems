"""Validate the multi-phase solver and bank the B5 triple-junction reference (CPU).

Rigorous checks:
  (1) N=2 grain shrinkage reproduces the single-phase curvature law R^2=R0^2-2*mu*sigma*t.
  (2) sum_alpha phi_alpha = 1 conserved to ~round-off.
  (3) a T-junction (initial 90/90/180) relaxes toward the equilibrium 120 degrees.

Self-contained: imports pf_solver + pf_solver_multiphase + NumPy only.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from pf_solver import curvature_law_radius, grain_radius
from pf_solver_multiphase import (MPParams, init_circle_2phase, init_t_junction,
                                  junction_angles, simulate, stable_dt)


def test_n2_curvature(N=128):
    dx = 1.0 / N
    p = MPParams(mu=0.02, sigma=1.0, eta=0.05)
    phis0 = init_circle_2phase(N, 1.0, 0.30, p.eta)
    t, frames, serr = simulate(phis0, p, dx, dt=stable_dt(p, dx), t_final=1.0,
                               save_every=max(1, int(0.05 / stable_dt(p, dx))))
    R = np.array([grain_radius(f[0], dx) for f in frames])  # phase 0 = grain
    m = R > 0.15
    slope = np.polyfit(t[m], R[m] ** 2, 1)[0]
    Rth = curvature_law_radius(t, 0.30, p.mu, p.sigma)
    return {"slope": float(slope), "slope_theory": -2 * p.mu * p.sigma,
            "slope_rel_err": abs(slope + 2 * p.mu * p.sigma) / (2 * p.mu * p.sigma),
            "max_radius_err": float(np.max(np.abs(R[m] - Rth[m]))),
            "max_sum_err": float(np.max(serr)), "R0": float(R[0]), "Rf": float(R[-1])}


def gen_triple_junction(N, outdir, mu=1.0, sigma=1.0, eta=0.05, t_final=0.06, n_frames=13):
    dx = 1.0 / N
    p = MPParams(mu=mu, sigma=sigma, eta=eta)
    dt = stable_dt(p, dx)
    phis0 = init_t_junction(N)
    t, frames, serr = simulate(phis0, p, dx, dt=dt, t_final=t_final,
                               save_every=max(1, int(np.ceil(t_final / dt / n_frames))))
    idx = np.linspace(0, len(frames) - 1, n_frames).astype(int)
    phi_stack = np.stack([frames[i] for i in idx])  # [T, 3, N, N]
    tt = t[idx]
    angles = []
    for i in idx:
        a = junction_angles(frames[i])
        angles.append(None if a is None else sorted(a.tolist()))

    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(os.path.join(outdir, "ref_triple_junction.npz"),
                        phi=phi_stack, test_times=tt, sum_err=serr[idx])
    final_angles = angles[-1]
    meta = {"name": "triple_junction", "N": N, "mu": mu, "sigma": sigma, "eta": eta,
            "dx": dx, "dt": float(dt), "t_final": t_final, "n_phases": 3,
            "test_times": tt.tolist(), "max_sum_err": float(np.max(serr)),
            "angles_over_time": angles, "final_angles_deg": final_angles}
    json.dump(meta, open(os.path.join(outdir, "ref_triple_junction.json"), "w"),
              indent=2, default=float)
    return meta


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--outdir", default=os.path.join(here, "ref_data"))
    ap.add_argument("--N", type=int, default=128)
    a = ap.parse_args()

    print("=== (1) N=2 reduces to single-phase curvature law ===")
    r = test_n2_curvature(a.N)
    print(f"  d(R^2)/dt={r['slope']:.4f} (theory {r['slope_theory']:.4f}, "
          f"rel_err={r['slope_rel_err'] * 100:.2f}%)  max|R-Rth|={r['max_radius_err']:.3e}")
    print(f"  max sum-constraint error = {r['max_sum_err']:.2e}  (R {r['R0']:.4f}->{r['Rf']:.4f})")

    print("=== (2,3) triple junction: sum conservation + angle relaxation ===")
    m = gen_triple_junction(a.N, a.outdir)
    print(f"  max sum-constraint error = {m['max_sum_err']:.2e}")
    print(f"  junction angles (deg): initial={m['angles_over_time'][0]} -> "
          f"final={m['final_angles_deg']}  (equilibrium = 120/120/120)")
    json.dump(r, open(os.path.join(a.outdir, "validate_n2_curvature.json"), "w"),
              indent=2, default=float)
    print("done; datasets in", a.outdir)


if __name__ == "__main__":
    main()
