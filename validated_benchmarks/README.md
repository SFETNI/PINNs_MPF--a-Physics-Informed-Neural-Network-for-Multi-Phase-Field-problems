# Validated Benchmarks — Evidence

This directory holds the **curated results** for each PINNs-MPF benchmark: figures, metrics,
reference data, selected weights, and a per-benchmark README. These are the *outputs*.

The *runnable scripts* that produce them live in [`../benchmarks/`](../benchmarks/), which
drives the [`../pinns_mpf/`](../pinns_mpf/) library. The two directories are complementary,
not duplicates:

- **`benchmarks/`** — runnable entry points (`run.py`, `run_decomp.py`, `run_multiNN.py`) that
  *generate* the evidence.
- **`validated_benchmarks/`** (here) — the *generated* evidence, self-contained per benchmark,
  with figures that regenerate on CPU from packaged data (no training).

| Benchmark | What it shows |
|---|---|
| [B1_traveling_wave](B1_traveling_wave/) | analytic moving-interface baseline (high precision) |
| [B2_single_network_driving_force](B2_single_network_driving_force/) | why decomposition is needed — the single-network limit (a **motivating baseline, not a validated result**) |
| [B3_driven_force](B3_driven_force/) | driven-force grain shrinkage, nine-worker time-marched with a 3×3 non-uniform spatial decomposition |
| [B4_grain_shrinkage](B4_grain_shrinkage/) | curvature-driven grain shrinkage |
| [B5_triple_junction](B5_triple_junction/) | four-phase triple-junction relaxation (128×128) |
| B6 multi-grain scaling | *exploratory study, not a validated benchmark* — see [`../explorations/multi_grain_scaling/`](../explorations/multi_grain_scaling/) |

See the top-level [README](../README.md) for the visual gallery and the
[Technical Report](../TECHNICAL_REPORT.md) for the governing physics and loss formulation.
