"""Generate & bank phase-field reference datasets (CPU / NumPy only).

Self-contained (imports only ``pf_solver`` + NumPy) so it can run on a GPU-less,
offline box. Produces ground-truth grain-shrinkage fields for the 2D benchmarks
and validates the no-driving-force case against the analytic curvature law.

    python generate_references.py            # default scenarios -> ./ref_data/
    python generate_references.py --outdir /path
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from pf_solver import (PFParams, curvature_law_radius, grain_radius, init_circle,
                       simulate, stable_dt)


def generate(name, mu, sigma, eta, delta_g, R0, N, t_final, n_test_times, outdir):
    dx = 1.0 / N
    phi0, xs = init_circle(N, 1.0, R0, eta)
    p = PFParams(mu=mu, sigma=sigma, eta=eta, delta_g=delta_g)
    dt = stable_dt(p, dx)
    times, frames, radii = simulate(phi0, p, dx, dt=dt, t_final=t_final,
                                    save_every=max(1, int(0.01 / dt)))
    test_times = np.linspace(0.0, t_final, n_test_times)
    idx = [int(np.argmin(np.abs(times - tt))) for tt in test_times]
    phi_stack = np.stack([frames[i] for i in idx])          # [T, N, N]
    R = np.array([grain_radius(frames[i], dx) for i in idx])

    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(os.path.join(outdir, f"ref_{name}.npz"),
                        phi=phi_stack, test_times=test_times, R=R, xs=xs)
    meta = dict(name=name, mu=mu, sigma=sigma, eta=eta, delta_g=delta_g, R0=R0,
                N=N, dx=dx, dt=float(dt), t_final=t_final, nsteps=len(times),
                test_times=test_times.tolist(), R=R.tolist())

    note = ""
    if delta_g == 0.0:
        Rth = curvature_law_radius(test_times, R0, mu, sigma)
        m = R > 0.15
        meta["curvature_law_max_abs_err"] = float(np.max(np.abs(R[m] - Rth[m])))
        note = f"curvature-law max|R-Rth|={meta['curvature_law_max_abs_err']:.3e}"
    json.dump(meta, open(os.path.join(outdir, f"ref_{name}.json"), "w"),
              indent=2, default=float)
    print(f"[{name}] R {R[0]:.4f} -> {R[-1]:.4f}  ({len(test_times)} frames, "
          f"{N}x{N}, dg={delta_g})  {note}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--outdir", default=os.path.join(here, "ref_data"))
    ap.add_argument("--N", type=int, default=128)
    a = ap.parse_args()
    # B2/B4-type single-phase grain shrinkage references
    generate("curvature", 0.02, 1.0, 0.05, 0.0, 0.30, a.N, 1.0, 11, a.outdir)
    generate("driving_force", 0.02, 1.0, 0.05, -0.3, 0.30, a.N, 1.0, 11, a.outdir)
    print("done; datasets in", a.outdir)


if __name__ == "__main__":
    main()
