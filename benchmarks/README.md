# Benchmarks — Runnable Scripts

Runnable benchmark entry points for the PINNs-MPF package. These scripts **produce** the
curated evidence in [`../validated_benchmarks/`](../validated_benchmarks/); this directory is
the code, that one is the results.

## Entry points by benchmark

| Benchmark | Entry point | Notes |
|---|---|---|
| **B1** traveling wave | `b1_traveling_wave/run.py` | analytic baseline |
| **B2** single-network limit | `b2_grain_shrinkage/run.py` | single-network baseline demonstration (see `../validated_benchmarks/B2_single_network_driving_force/`) |
| **B3** driven-force shrinkage | `b3_driven_force/run.py` | wrapper over the shared engine; default option `B3_DRIVEN_FORCE` is the validated config (see below) |
| **B4** curvature-driven shrinkage | `b4_grain_shrinkage/run.py` | wrapper over the shared engine (options `B4N_*`) |
| **B5** triple junction | `b5_triple_junction/run_multiNN.py` | MultiNN / time-marched triple junction |
| **B6** multi-grain scaling | *exploratory* | included as an exploratory study under `../explorations/multi_grain_scaling/` (self-contained CPU figure scripts; no training entry point here) |

### Shared grain-shrinkage engine

B3 and B4 are both produced by one engine, `b2_grain_shrinkage/run_decomp.py` (the
`b2_grain_shrinkage/` folder name is historical, kept to preserve the provenance manifest).
The `b3_driven_force/` and `b4_grain_shrinkage/` folders are thin wrappers that select the
right option family (`B3*` / `B4N*`) on that engine. `b2_grain_shrinkage/run.py` is the
single-network runner used for the B2 baseline.

### Reproducing the validated B3 result

The engine's default B3 option, `B3_DRIVEN_FORCE`, **is** the validated configuration
(`Δg = −0.5`, `R₀ = 38`, `128×128`, `3×3` non-uniform workers with core edges `[0, 20, 108, 128]`
and `overlap = 7`, held by the per-window Adam↔L-BFGS convergence gate to `loss < 2e-6` with
best-cycle restore, marched `K = 40` windows to `t = 46`):

```bash
# reproduce the validated benchmark (GPU recommended; float64 is the default)
python benchmarks/b3_driven_force/run.py --outdir outputs/b3_driven_force
# or inspect the exact resolved config without training:
python benchmarks/b3_driven_force/run.py --print-config
```

The older `B3` / `B3_SMOKE` options are the **exploratory** strong-driving case (`Δg = −250`,
2×2 workers, single Adam→L-BFGS pass, no convergence gate) — a documented dead-end kept for
provenance, not the default. See `b3_driven_force/README.md`.

### Other diagnostic paths

- `b5_triple_junction/run.py` — single-network multiphase diagnostic; not the accepted B5
  architecture (use `run_multiNN.py`).

Long GPU runs should be launched only after review of the intended configuration.
