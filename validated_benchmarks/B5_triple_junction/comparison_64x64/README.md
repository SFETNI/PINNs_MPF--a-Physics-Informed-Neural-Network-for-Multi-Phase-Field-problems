# 64 x 64 Triple-Junction Reference Comparison

This folder retains the earlier `64 x 64` triple-junction validation as a comparison point for the current `128 x 128` result.

The 64 x 64 case demonstrates the same PINNs-MPF time-window protocol on a smaller grid and a different, near-equilibrium initial condition. It is retained as a compact comparison within the same triple-junction benchmark family, while the main B5 result uses the `128 x 128` 90-degree relaxation.

## Files

- `figures/triple_junction_reference_vs_pinns_mpf_64x64.png` - finite-difference reference, PINNs-MPF prediction, and difference maps
- `figures/triple_junction_metrics_64x64.png` - validation metrics
- `media/triple_junction_evolution_64x64.gif` - visual evolution
- `media/triple_junction_relaxation_64.gif` - smooth animation of this 64x64 finite-difference reference (near-equilibrium initial condition), with junction-angle and interfacial-energy readouts (companion layout to the 128x128 relaxation animation)
- `make_triple_junction_evolution_64.py` - regenerator for the 64x64 relaxation animation (reads `reference/reference_64x64_t200.npz`)
- `reference/reference_64x64_t200.npz` - finite-difference reference trajectory
- `reference/reference_64x64_t200_metadata.json` - reference metadata
- `run/summary_64x64_t200.json` - run summary
