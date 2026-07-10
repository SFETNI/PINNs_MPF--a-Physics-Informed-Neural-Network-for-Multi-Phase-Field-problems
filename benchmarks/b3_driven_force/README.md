# B3 — Driven-Force Grain Shrinkage (runner)

Runnable entry point for B3. B3 uses the shared multi-network / time-marched grain-shrinkage
engine in [`../b2_grain_shrinkage/run_decomp.py`](../b2_grain_shrinkage/run_decomp.py)
(driven force, `Δg ≠ 0`); `run.py` here is a thin wrapper that delegates to it.

## Reproduce the validated benchmark

The default option `B3_DRIVEN_FORCE` **is** the validated configuration: corrected-scale
physics `Δg = −0.5` (Péclet `Pe = |Δg|·R₀/σ = 19`, moderate driving), grain `R₀ = 38` on a
`128×128` grid, `3×3` **non-uniform** workers with core edges `[0, 20, 108, 128]` and
`overlap = 7` (one large central worker holds the grain, thin outer workers hold the matrix),
held by the per-window legacy convergence protocol — repeat `[Adam(3000) → L-BFGS(500)]` up to
`cycles = 8` times per window, advancing only once the total weighted loss drops below
`loss_thresh = 2e-6`, and keeping the best-loss cycle's weights (`restore_best_cycle = True`).
The march runs `K = 40` uniform windows (`dt = 1.15`) to `t = 46` with exact FD
window-boundary alignment (`fd_save_every = 1`).

```bash
# Reproduce the validated B3 driven-force benchmark (GPU recommended; float64 is the default).
python benchmarks/b3_driven_force/run.py --outdir outputs/b3_driven_force

# Inspect the exact resolved configuration without training (CPU, no optimizer / no FD):
python benchmarks/b3_driven_force/run.py --print-config

# Optional: march only the first N windows for a quick check (e.g. the early handoff at t≈6.9):
python benchmarks/b3_driven_force/run.py --outdir outputs/b3_early --handoff-at 6
```

Precision: run in `float64` (the default) for the validated result — the per-window L-BFGS
polish needs it. `fp32` is a faster screening mode (`PINNS_MPF_DTYPE=float32`) but does not
reach the same high-precision floor.

## Exploratory option (not the validated result)

`B3` and `B3_SMOKE` are a documented dead-end kept for provenance only: strong-driving
`Δg = −250`, `2×2` workers, a single `Adam → L-BFGS` pass, and **no** convergence gate. They
do not reproduce the validated benchmark and are not the default.

```bash
# exploratory strong-driving case — NOT the validated result
python benchmarks/b3_driven_force/run.py --option B3 --outdir <dir>
```

Curated evidence and figures:
[`../../validated_benchmarks/B3_driven_force/`](../../validated_benchmarks/B3_driven_force/).
