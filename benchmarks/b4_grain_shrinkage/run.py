#!/usr/bin/env python
"""B4 - curvature-driven grain shrinkage (entry point).

B4 is produced by the shared multi-network / time-marched grain-shrinkage engine in
``../b2_grain_shrinkage/run_decomp.py`` (curvature-driven, Delta_g = 0). This thin wrapper
delegates to that engine, restricting the option list to the curvature-driven family.

The validated trajectory is assembled from three intervals, each a run of that engine:

    early  : --option B4N_CAP_RAMP
    middle : --option B4N_GEOM_CENTER
    late   : --option B4N_GEOM_EXTEND

Curated evidence: ../../validated_benchmarks/B4_grain_shrinkage/

Usage:
    python benchmarks/b4_grain_shrinkage/run.py [--option B4N_CAP_RAMP] [--outdir DIR] [--quick]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from benchmarks.b2_grain_shrinkage.run_decomp import run, OPTIONS  # noqa: E402

_B4_OPTIONS = [k for k in sorted(OPTIONS) if k.startswith("B4")]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="B4 curvature-driven grain shrinkage")
    ap.add_argument("--option", default="B4N_CAP_RAMP", choices=_B4_OPTIONS,
                    help="curvature-driven option (validated: B4N_CAP_RAMP, "
                         "B4N_GEOM_CENTER, B4N_GEOM_EXTEND)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    run(quick=a.quick, outdir=a.outdir, option=a.option)
