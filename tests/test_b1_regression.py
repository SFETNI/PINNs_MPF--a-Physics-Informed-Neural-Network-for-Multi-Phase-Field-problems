"""Regression test: Benchmark 1 (traveling wave) vs analytic solution.

Runs the fast ``quick`` config (~2 min on the A10, float64) and asserts the PINN
matches the exact solution (Eq. 26). Thresholds sit ~1-2 orders above the observed
fast-config result (MSE ~2e-6, v_n error ~1%) so the test is robust but meaningful;
the high-precision run reaches MSE ~3e-7.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmarks.b1_traveling_wave.run import run  # noqa: E402


def test_b1_quick_regression(tmp_path):
    # tmp_path keeps the test from overwriting the stored B1 artifacts
    m = run(quick=True, make_plots=False, outdir=str(tmp_path))
    # Primary high-precision criterion (observed ~2e-6; paper target band is 1e-4..1e-6).
    assert m["mse"] < 1e-4, f"MSE too high: {m['mse']:.3e}"
    # Secondary physics sanity check. The fast config under-converges vs the
    # high-precision run (which reaches 0.04%); GPU float64 is mildly non-deterministic
    # run-to-run, so the velocity (a phi=0.5 crossing fit) carries ~1-2% noise here.
    assert m["v_n_rel_err"] < 0.05, f"interface velocity off: {m['v_n_fit']} vs {m['v_n_true']}"
