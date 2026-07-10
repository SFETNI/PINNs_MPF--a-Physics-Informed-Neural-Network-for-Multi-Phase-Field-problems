# Legacy Version — Original Published PINNs-MPF Code

This directory preserves the **original code** that accompanied the published article, for
lineage and reference:

> Elfetni & Darvishi Kamachali, *PINNs-MPF: A Physics-Informed Neural Network framework for
> Multi-Phase-Field simulation of interface dynamics*, Engineering Analysis with Boundary
> Elements **176** (2025) 106200. [DOI](https://doi.org/10.1016/j.enganabound.2025.106200)

The authors' own top-level notes are kept here as [`README_original.md`](README_original.md).

## Code only

To keep the package small and to let readers focus on the current implementation and results,
this copy contains **source code only**. Removed from the original working tree:

- figures and images (`.png`, `.jpg`, `.gif`),
- saved model weights and generated data (the per-worker `.json` weight dumps, `.npz`, `.npy`,
  `.pkl`, `.mat`, `.db`),
- runtime logs (loss logs, `usage_log.txt`, …), and notebook cell outputs (cleared).

What remains is the Python sources (`.py`), the reference/analysis notebooks (`.ipynb`, outputs
cleared), the original per-benchmark `README.md` files, and run scripts (`.sh`) — about 3 MB.
The code is provided as a historical record; it is the original TensorFlow/CPU implementation
and is **not maintained or runnable as-is** here. The current, consolidated, GPU-native,
re-validated implementation is the rest of this package (`../pinns_mpf/`, `../benchmarks/`,
`../validated_benchmarks/`).

## The original framework

From the article abstract: the domain is subdivided into multiple batches, each associated with
an independent neural network, and a **Master NN** handles the interaction among the networks
and the transfer of learning across space, time, and phases. The benchmarks below exercise that
framework on problems of increasing complexity.

## Benchmark map (original → current)

| Original folder | Physics | Current benchmark |
|---|---|---|
| `Benchmark_1_travelling_wave_interface/` | analytic moving interface (1 NN) | **B1** traveling wave |
| `Benchmark_2_dual_phase__one_phase_1NN_driving_force/` | driven-force grain shrinkage, single network | **B2** single-network limit |
| `Benchmark_3_dual_phase_one_phase_4NN_driving_force/` | driven-force grain shrinkage, 4 NN | **B3** driven-force grain shrinkage |
| `Benchmark_4_dual_phase_one_phase_4NN_interfacial_motion/` | curvature / interfacial motion, 4 NN | **B4** curvature-driven grain shrinkage |
| `Benchmark_5_triple_junction_16NN_main/` | four-phase triple junction (16 NN) | **B5** triple junction |
| `Benchmark_6__triple_junction_16NN_Pyramidal_training/` | triple junction, pyramidal training | **B6** — see the exploratory scaling study in `../explorations/multi_grain_scaling/` |
| `Supplementary/` | phase-field reference solver, theory, post-processing | reference material |

See the top-level [README](../README.md) for the current results and
[CITATION.cff](../CITATION.cff) for how to cite the work.
