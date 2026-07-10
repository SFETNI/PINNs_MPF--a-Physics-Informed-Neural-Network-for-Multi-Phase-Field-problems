# B1 Traveling Wave

Status: validated analytic baseline.

Reference: closed-form traveling interface solution.

Main metrics:

- MSE: `2.675e-7`
- MAE: `1.545e-4`
- relative L2: `6.898e-4`
- fitted velocity: `0.499818`
- true velocity: `0.5`

Contents:

- `figures/solution.png`
- `figures/loss.png`
- `data/metrics.json`
- `data/weights.npz`

This benchmark establishes that PINNs-MPF reaches high precision on an analytic interface evolution problem.
