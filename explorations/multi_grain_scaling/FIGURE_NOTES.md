# Figure notes — scaling study (capability envelope & method options)

Per-figure sources, what each figure shows, and its scope. All figures regenerate CPU-only from
`data/` and `metrics/` in this directory (no external run directories).

---

### `b6_capability_fd_vs_pinn.png` — capability envelope

- **Source:** `data/fd_reference_marched.npz` (reference frames/times), `data/initial_condition.npz`
  (t=0), `data/pinns_mpf_direct_final_field.npz` (reference-selected marched endpoint),
  `metrics/direct_capability_metrics.json` (per-window PINNs-MPF vs persistence). Built by
  `scripts/make_figures.py::capability_hero()`.
- **Shows:** the reference / PINNs-MPF / |difference| fields at t=0 and t=100, plus two panels
  comparing per-window topology and field error against the persistence (hold-the-handoff-field)
  baseline, and a footer with the whole-horizon comparison against holding the t=0 field.
- **Scope:** an accuracy characterisation, not a field-accuracy validation. The trajectory uses
  reference handoff at each window (re-anchored to the reference every 25 time units) and the
  displayed field is reference-selected among physically valid checkpoints; both are marked on the
  figure and should be read together with the field panels.

### `b6_fd_relaxation.png` — reference context

- **Source:** `data/fd_reference_marched.npz`, all five window-boundary frames.
  `scripts/make_figures.py::relaxation_strip()`.
- **Shows:** the finite-difference reference alone (90-degree brick → 120-degree honeycomb).
- **Scope:** physical context for the benchmark; it carries no PINNs-MPF result.

### `b6_pyramid_concept_w1.png` — how the decomposition applies (geometry)

- **Source:** `data/initial_condition.npz` (grains, phase coloring, diffuse field) and the recorded
  child selection in `metrics/pyramid_w1_metrics.json`. `scripts/make_pyramid_figures.py::concept_field()`.
- **Shows:** the window-1 geometry, in six panels — (a) the 16-grain microstructure, (b) its mapping
  to C=6 softmax channels, (c) the diffuse initial field (dark = interfaces/junctions, width ~η), and
  (d–f) the three pyramid levels (4×4 → 2×2 → 1×1 workers). Each level marks the triple junctions
  (white), the junctions-per-worker load (numbers, J/W mean in the title), the 2×2 parent seam (red),
  and — at the 4×4 level — the interface-richest fine box selected to seed each parent quadrant (green).
- **Scope:** concept and geometry only; it carries no performance numbers. It illustrates how the
  fine-to-coarse decomposition and interface-based child selection are laid out on this problem; the
  quantitative accuracy/health/cost comparison is in `b6_pyramid_vs_2x2_metrics_losses.png`.

### `b6_pyramid_vs_2x2_metrics_losses.png` — pyramidal vs flat 2×2

- **Source:** `metrics/pyramid_w1_metrics.json` (parent fine-tuning) and
  `metrics/direct_baseline_metrics.json` (flat-2×2 window 1). `::compare_to_direct()`.
- **Shows:** reference agreement, topology, health, and all unweighted loss terms — the pyramidal
  path (with fine-box cost on the x-axis) against a flat 2×2 decomposition. The persistence baseline
  is drawn; both paths approach it and then rise above it as training continues.
- **Scope:** a single window and single configuration; the two paths reach comparable accuracy at
  comparable cost here, and the figure does not extend to other grain counts, deeper pyramids, or 3D.

### `b6_pyramid_w1_convergence.png` — supplementary

- **Source:** `metrics/pyramid_w1_metrics.json` (parent fine-tuning). `::convergence()`.
- **Shows:** parent refinement after transfer is smooth and monotone, and the seam is repaired.
- **Scope:** window 1 of this configuration only; an optional fifth figure.

---

## Reproducibility

- Both scripts run CPU-only and self-contained from `data/` and `metrics/` (no external run
  directories); all five figures regenerate.
- All five figures were rendered and inspected after generation.
- The raw `metrics/*.json` retain solver keys for provenance; the reader-facing documents
  and figures use plain scientific terms only.
