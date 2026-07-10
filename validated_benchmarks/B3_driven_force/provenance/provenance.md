# Provenance — Driven-Force Grain Shrinkage Package

**Assembled:** 2026-07-07 (deterministically from packaged data; no network fetch).

## What this package contains

Reader-facing evidence for the driven-force grain-shrinkage benchmark: the accepted radius metrics, a
compact summary, publication-style figures, and an animation. Everything is derived from an existing
completed run; no training, optimization, or reference generation was performed to assemble it.

## Source

- Time-marched PINNs-MPF run on an NVIDIA A10 GPU, TensorFlow 2.16.2, `float64`.
- Domain `128 × 128` (`dx = 1`), `μ = σ = 1`, `η = 7` cells, driving force `Δg = −0.5`, initial radius
  `R₀ = 38` cells, 40 uniform time intervals of `dt = 1.15` (`t = 0 → 46`).
- Reference: explicit finite-difference phase-field solver evaluated on the same grid; the reference
  radius aligns to machine precision (`fd_boundary_max_abs_error ≈ 7e-15`) at each interval boundary.
- Confirmation was run fresh to the interval-20 target; the same run was then continued to interval 40
  (near extinction).

## How the figures were made

- `make_figures.py` (CPU only, `CUDA_VISIBLE_DEVICES=""`) rebuilds every figure and the animation from
  `data/driven_force_metrics.json` and the packaged phase-field snapshots.
- The finite-difference reference is shown as a radius (overlaid circle / curve); no finite-difference
  field was regenerated for this package.
- No finite-difference field, radius law, or reference snapshot was used as a training label at any point
  in the underlying run (reference-free after each interval's initial condition).

## Integrity

`checksums.sha256` lists SHA-256 digests for the packaged data, figures, media, README, and figure script.
Verify with:

```bash
cd ..            # package root
sha256sum -c provenance/checksums.sha256
```
