#!/usr/bin/env python
"""B3 - driven-force grain shrinkage (entry point).

B3 is produced by the shared multi-network / time-marched grain-shrinkage engine in
``../b2_grain_shrinkage/run_decomp.py`` (driven force, Delta_g != 0). This thin wrapper
delegates to that engine.

The DEFAULT option ``B3_DRIVEN_FORCE`` is the VALIDATED configuration: corrected-scale
physics Delta_g = -0.5 (Pe = |Delta_g|*R0/sigma = 19, moderate driving), grain R0 = 38 on
a 128x128 grid, 3x3 NON-UNIFORM workers with core edges [0, 20, 108, 128] and overlap = 7,
held by the per-window legacy convergence protocol (repeat [Adam(3000) -> L-BFGS(500)] up
to cycles = 8 times per window, advancing only once the total weighted loss < 2e-6, keeping
the best-loss cycle's weights), marched K = 40 uniform windows (dt = 1.15) to t = 46 with
exact FD window-boundary alignment. It reproduces the evidence in
``../../validated_benchmarks/B3_driven_force/``. Run in float64 (the default) for the
validated result.

The ``B3`` / ``B3_SMOKE`` options are EXPLORATORY ONLY (strong-driving Delta_g = -250, 2x2
workers, single Adam->L-BFGS pass, no convergence gate) — a documented dead-end kept for
provenance, not the validated result.

Usage:
    # Reproduce the validated B3 driven-force benchmark (GPU recommended; float64):
    python benchmarks/b3_driven_force/run.py --outdir outputs/b3_driven_force

    # Inspect the resolved configuration without training (CPU, no optimizer / no FD):
    python benchmarks/b3_driven_force/run.py --print-config

    # Exploratory strong-driving dead-end (NOT validated):
    python benchmarks/b3_driven_force/run.py --option B3 --outdir DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from benchmarks.b2_grain_shrinkage.run_decomp import run, OPTIONS  # noqa: E402

_B3_OPTIONS = [k for k in sorted(OPTIONS) if k.startswith("B3")]
_EXPLORATORY = {"B3", "B3_SMOKE"}
_VALIDATED = "B3_DRIVEN_FORCE"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="B3 driven-force grain shrinkage (default = validated B3_DRIVEN_FORCE)")
    ap.add_argument("--option", default=_VALIDATED, choices=_B3_OPTIONS,
                    help="benchmark option (default: the validated B3_DRIVEN_FORCE)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--handoff-at", type=int, default=None,
                    help="stop after this many windows (partial march, e.g. 6 or 20)")
    ap.add_argument("--resume", action="store_true",
                    help="resume a partial march from the last checkpoint in --outdir")
    ap.add_argument("--print-config", action="store_true",
                    help="print the resolved option config as JSON and exit (no training, no FD)")
    a = ap.parse_args()

    if a.print_config:
        print(json.dumps({"option": a.option, "config": OPTIONS[a.option]}, indent=2, default=str))
        sys.exit(0)

    if a.option in _EXPLORATORY:
        print("\n[NOTE] This runs an EXPLORATORY driven-force option (Delta_g=-250), which is NOT\n"
              "       the validated B3 configuration. Use the default (B3_DRIVEN_FORCE) to\n"
              "       reproduce the validated result. See this benchmark's README.\n")
    else:
        print(f"\n[B3] Running the VALIDATED driven-force option '{a.option}'\n"
              "     (Delta_g=-0.5, Pe=19, R0=38, 128x128, 3x3 workers, convergence-gated to loss<2e-6).\n")

    run(quick=a.quick, outdir=a.outdir, seed=a.seed, option=a.option,
        resume=a.resume, handoff_at=a.handoff_at)
