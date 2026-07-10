# B5 Triple Junction Validation Summary

This summary documents the accepted `128 x 128` triple-junction validation through `t=200` (`2.28 tau_c`).

## Numerical Setting

- Reference: finite-difference multiphase-field simulation
- Grid: `128 x 128`
- Parameters: `eta=6/64`, `mu=1e-4`, `sigma=1`
- Time windows: eight windows of `dt=25`
- PINNs-MPF architecture: four spatial workers, each with four softmax phase outputs
- Training: Adam, 800 steps per window
- Handoff: each window starts from the phase-field reference at the same time

## Result

The first window is a rapid transient. The finite-difference reference rounds the junctions strongly by `t=25`, while PINNs-MPF still retains part of the rectangular initial geometry. This is the disclosed limitation of the validation.

From `t=50` onward, PINNs-MPF tracks the finite-difference reference closely. The final window at `t=200` reaches:

- MSE vs reference: `3.70e-4`
- Argmax disagreement: `1.36%`
- Junction count: `6`
- Angle-tracking gap: `0.40 deg`
- Final bottom-top seam mismatch: `0.0181`
- Maximum bottom-top seam mismatch over all windows: `0.0214`
- Phase-sum error: `3.33e-16`

The accepted claim is field and topology tracking under the time-window protocol, not a fully autonomous rollout from only the global `t=0` field.
