# Provenance and build log — B6 scaling study (capability envelope & method options)

**date:** 2026-07-09 · **status:** integrated under `explorations/` (exploratory tier), CPU-only,
no GPU used · **outside the validated-benchmark set (B1–B5).**

---

## 1. What this is

A self-contained, CPU-only **exploratory framework study**: how PINNs-MPF behaves on a larger,
harder microstructure (16 grains / 6 phases / N=256, ~32 junctions) than the validated benchmarks,
plus a demonstration of the optional pyramidal-initialization mechanism. It is **not** a validated
benchmark. All inputs are shipped under `data/` + `metrics/`; the figures regenerate with

```bash
CUDA_VISIBLE_DEVICES="" python scripts/make_figures.py          # capability + relaxation
CUDA_VISIBLE_DEVICES="" python scripts/make_pyramid_figures.py  # pyramid concept / compare / convergence
```

No GPU, no training, no network inference. sha256 + sizes for every file are in the repository-root
`provenance/MANIFEST.json`.

## 2. Provenance — source runs

Each shipped file is a copy of an output from one of three source runs on this configuration
(N=256, 16 grains, 4×4 fine decomposition): a marched **direct** run, a flat **2×2 baseline** run,
and a **pyramidal-initialization** run.

| shipped file | source |
|---|---|
| `data/fd_reference_marched.npz` | marched finite-difference reference (K=16, N=256, 4×4); identical across the three runs, shipped once |
| `data/initial_condition.npz` | the designed 16-grain brick initial condition (N=256, 4×4) |
| `data/pinns_mpf_direct_final_field.npz` | marched direct run — final field selected from physically valid checkpoints |
| `data/pinns_mpf_pyramid_w1_field.npz` | pyramidal-initialization run — window-1 parent field |
| `metrics/direct_capability_metrics.json` | marched direct run — metrics |
| `metrics/direct_baseline_metrics.json` | flat 2×2 baseline run — metrics |
| `metrics/pyramid_w1_metrics.json` | pyramidal-initialization run — metrics |

The capability run is the marched direct run using reference-based selection among physically valid
checkpoints; the pyramid and flat-2×2 runs are the window-1 method-option comparison.

## 3. Key numbers (from the shipped metrics)

- Capability, whole horizon (t=0→100): PINNs-MPF endpoint **argmax 4.0% / MSE 3.4e-3** vs holding
  the t=0 field (persistence) **7.1% / 8.7e-3** — the prediction is closer to the reference than the
  unevolved initial condition over the horizon.
- Capability, per window (reference handoff): the evolved field is not closer to the reference than
  the handed field on MSE in any of the 4 windows.
- Pyramid W1 saved field: **MSE 1.116e-2 / argmax 9.4%**; lowest checkpoint seen 6.45e-3; persistence
  5.75e-3 — on this window the pyramid field is not closer to the reference than persistence; the
  figure illustrates the selection/transfer/repair mechanism.

## 4. Scope

- **Tier.** This study lives under `explorations/`, distinct from `validated_benchmarks/`. It reports
  a scaling study rather than a validation of the kind reported for B1–B5.
- **Reference dependence.** The capability result uses reference-handoff marching and
  reference-based checkpoint selection — marked on the figure and in the README; the trajectory is
  re-anchored to the reference at each window rather than rolled out autonomously.
- **Pyramid.** On this configuration the pyramidal and flat 2×2 paths reach comparable accuracy at
  comparable cost; the technique is presented as an open direction for convergence-limited regimes.
- **Provenance JSON.** `metrics/*.json` retain raw solver keys for provenance and are not
  intended as reader-facing text.
- No frozen/validated baselines or `pinns_mpf/` files were touched; no GPU; no network access.
