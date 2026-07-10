# Reader-facing media — space–time decomposition (Figure A, animated)

Hero visual for the PINNs-MPF README. Animated counterpart of the static
"Figure A" schematic.

| File | What it is |
|---|---|
| `pinns_mpf_spacetime.gif` | Animated hero — 1056×748, ~14.5 s loop @ 15 fps, ~6.3 MB. Fixed camera, eased pacing with dwell/holds; the active window is a solid 3D slab and a left rail tracks the W₁→W_N march. A **fixed** legend sits on the right; a **scalable-decomposition inset** (bottom-right) cycles 2×2 → 4×4 → 6×6. |
| `pinns_mpf_spacetime.png` | Static fallback (final full-stack frame) for viewers that don't play GIFs |
| `make_spacetime_animation.py` | Regenerator for the animation. `python make_spacetime_animation.py out.gif` (needs matplotlib, numpy, pillow). Dump one test frame: `... x 150`. |
| `pinns_mpf_workflow.png` | Static **"Implemented Method"** infographic (Figure B) — a 5-stage softmax-MultiNN pipeline (window setup → sampling → spatial MultiNN → softmax + PI loss → output + validation), illustrated on the **real** B5 triple-junction 90° → 120° relaxation. Grid size is deliberately not printed (the decomposition is illustrative and scales to any N×N). |
| `make_workflow_figure.py` | Regenerator for the infographic. `python make_workflow_figure.py`. Self-contained: it crops the real microstructure panels (90° initial field, PINNs-MPF prediction, FD reference, \|difference\|) from `validated_benchmarks/B5_triple_junction/figures/…128.png`, builds the interface-weighted sampling panel from the reference field, and reads the run's actual metrics — no generic art, no fabricated numbers. |

> **Representative, not literal.** Window labels are generic **W₁ … W_N** (the stack draws a
> representative 8) — they illustrate the space–time marching idea, **not** any single
> benchmark's exact window count. The **scalable-decomposition inset** advertises a real code
> capability: `make_boxes(L, nx, ny, …)` in `pinns_mpf/decomposition/decomp.py` builds `nx·ny`
> boxes, one worker network each, for **any** `nx×ny` (2×2, 3×3, 4×4, 5×5, … used across the
> B5 / B3 / B2 runners), plus non-uniform edges.

## What the animation shows (~14.5 s loop)

1. One periodic 2D domain appears.
2. The base plane splits into a **2×2 spatial decomposition** → 4 subdomains / **4 worker networks** (tiles 1–4).
3. The domain extrudes upward into **N time windows (W₁ → W_N)** — drawn as a representative stack of 8.
4. An **active window marches W₁ → W_N**: in each window the 4 worker networks act on their tiles,
   **internal continuity** (nodes on the interior cross) and the **periodic seam** (blue wrap arrows in x, y)
   are enforced. In autonomous use, a window-to-window handoff passes the predicted field up to the next
   window; validated protocols that use reference start fields should state that explicitly in the caption.
5. Ends on the full space–time stack.
6. Throughout, the **scalable-decomposition inset** cycles 2×2 → 4×4 → 6×6 to signal that the
   decomposition granularity is chosen to match the problem's complexity.

## Suggested README placement + caption

Place near the top, right after the one-line project summary:

```markdown
## How PINNs-MPF works

![PINNs-MPF space–time decomposition](pinns_mpf_spacetime.gif)

*PINNs-MPF space–time decomposition. The 2D domain is partitioned into four spatial
subdomains, and the evolution is solved over successive time windows. In each window,
one worker network handles each spatial batch, while internal continuity and periodic
seam conditions are enforced across interfaces. In autonomous marches, each window can
hand its predicted field off to the next; benchmark captions should disclose any
reference-field handoff used for validation.*
```

Follow it with the **static workflow infographic** (Figure B) under a "Implemented method"
heading, then the validation gallery (e.g. the 128×128 triple-junction result).

## Notes

- Pillow writes the GIF with lossless frame-differencing (static regions become deltas; hold
  frames are merged into longer per-frame delays — 206 stored frames, 14.5 s of playback). It
  plays correctly on GitHub and in browsers. To inspect a single true frame in ImageMagick you
  must `-coalesce` first, otherwise you see only that frame's delta.
- Reader-facing media assets used by the repository documentation.
