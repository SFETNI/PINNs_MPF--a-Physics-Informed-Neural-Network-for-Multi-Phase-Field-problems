# B4 — Curvature-Driven Grain Shrinkage (runner)

Runnable entry point for B4. B4 uses the shared multi-network / time-marched grain-shrinkage
engine in [`../b2_grain_shrinkage/run_decomp.py`](../b2_grain_shrinkage/run_decomp.py)
(curvature-driven, `Δg = 0`); `run.py` here is a thin wrapper that delegates to it.

The validated trajectory is assembled from three intervals:

| Interval | Option |
|---|---|
| early  | `B4N_CAP_RAMP` |
| middle | `B4N_GEOM_CENTER` |
| late   | `B4N_GEOM_EXTEND` |

```bash
python benchmarks/b4_grain_shrinkage/run.py --option B4N_CAP_RAMP --outdir <dir>
```

Curated evidence and figures:
[`../../validated_benchmarks/B4_grain_shrinkage/`](../../validated_benchmarks/B4_grain_shrinkage/).
