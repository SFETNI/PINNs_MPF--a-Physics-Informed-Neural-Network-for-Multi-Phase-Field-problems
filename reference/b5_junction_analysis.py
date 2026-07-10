"""Triple-junction analysis utilities for PINNs-MPF benchmarks."""
from __future__ import annotations

import numpy as np


def junction_angles_all(phis, L, margin_frac=0.10, min_score=0.01,
                        radii_frac=(0.05, 0.08, 0.10, 0.12, 0.15, 0.18)):
    """Detect ALL triple junctions and return per-junction angles + summary stats.

    Returns:
        junctions: list of dicts {ix, iy, x_phys, y_phys, angles, phases}
        stats: dict {n_detected, mean_angles, rms_dev_from_120}
        None, None if no junctions found
    """
    from scipy.ndimage import maximum_filter

    n = phis.shape[-1]
    s = np.sort(phis, axis=0)
    score = s[-1] * s[-2] * s[-3]

    m = max(1, int(margin_frac * n))
    masked = np.full_like(score, -1.0)
    masked[m:-m, m:-m] = score[m:-m, m:-m]

    # local maxima in a window of ~eta/3 cells
    win = max(3, n // 30)
    local_max = (masked == maximum_filter(masked, size=win)) & (masked > min_score)
    candidates = np.argwhere(local_max)

    # Non-maximum suppression: on a flat score plateau (common at coarse grids or
    # sharp/near-sharp interfaces) several adjacent cells tie for the local max and
    # each gets flagged, double-counting one physical junction. Greedily keep the
    # highest-score candidate and drop any other within merge_rad cells (periodic
    # distance on the torus). merge_rad << true junction spacing, so distinct
    # junctions are never merged.
    if len(candidates) > 1:
        cand_scores = masked[candidates[:, 0], candidates[:, 1]]
        order = np.argsort(cand_scores)[::-1]
        merge_rad = max(win, int(0.06 * n))
        kept = []
        for idx in order:
            ix, iy = int(candidates[idx, 0]), int(candidates[idx, 1])
            if all(min(abs(ix - kx), n - abs(ix - kx)) > merge_rad or
                   min(abs(iy - ky), n - abs(iy - ky)) > merge_rad for kx, ky in kept):
                kept.append((ix, iy))
        candidates = np.array(kept)

    junctions = []
    for (ix, iy) in candidates:
        angles_local = _angles_at(phis, ix, iy, L, n, radii_frac)
        if angles_local is not None:
            dom3 = np.argsort(phis[:, ix, iy])[-3:]
            junctions.append({
                "ix": int(ix), "iy": int(iy),
                "x_phys": float(ix * L / n), "y_phys": float(iy * L / n),
                "angles": sorted(angles_local.tolist()),
                "dominant_phases": sorted(dom3.tolist()),
            })

    if not junctions:
        return None, None

    all_angles = np.array([j["angles"] for j in junctions])   # [J, 3]
    dev = np.abs(all_angles - 120.0)
    stats = {
        "n_detected": len(junctions),
        "mean_angles": sorted(all_angles.mean(axis=0).tolist()),
        "rms_dev_from_120": float(np.sqrt((dev ** 2).mean())),
        "max_dev_from_120": float(dev.max()),
    }
    return junctions, stats


def _angles_at(phis, ix, iy, L, n, radii_frac):
    """Measure sector angles around a specific grid point (ix, iy)."""
    thetas = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    for rf in radii_frac:
        rad = max(2, int(rf * n))
        ii = (np.rint(ix + rad * np.cos(thetas)).astype(int)) % n
        jj = (np.rint(iy + rad * np.sin(thetas)).astype(int)) % n
        dom = np.argmax(phis[:, ii, jj], axis=0)
        ch = np.where(dom != np.roll(dom, 1))[0]
        unique_phases = set(dom.tolist())
        if len(unique_phases) == 3 and len(ch) == 3:
            bnd = np.sort(thetas[ch])
            gaps = np.diff(np.concatenate([bnd, [bnd[0] + 2 * np.pi]]))
            return np.degrees(gaps)
    return None


# ── article IC loader ────────────────────────────────────────────────────────

def load_article_ic(target_N):
    """Load legacy B5 IC (65×65) and resample to target_N×target_N.

    The legacy IC uses a 65-point endpoint-inclusive grid (x∈[0,1], 65 pts),
    matching Nx=65, lb=[0,0], ub=[1,1] in the paper code. We resample to
    target_N points on [0,1) (endpoint=False) for the periodic FD solver.
    """
    from scipy.interpolate import RegularGridInterpolator

    d = np.load(LEGACY_IC_PATH)
    phi_raw = d["all_phases"].astype(np.float64)   # (4, 65, 65)
    n_raw = phi_raw.shape[-1]                       # 65

    if target_N == n_raw - 1:
        # Convenient: just drop the repeated endpoint
        phi = phi_raw[:, :target_N, :target_N]
    else:
        # General interpolation
        xs_raw = np.linspace(0.0, 1.0, n_raw, endpoint=True)
        xs_new = np.linspace(0.0, 1.0, target_N, endpoint=False)
        phi = np.zeros((4, target_N, target_N), dtype=np.float64)
        for k in range(4):
            interp = RegularGridInterpolator(
                (xs_raw, xs_raw), phi_raw[k], method="linear",
                bounds_error=False, fill_value=None)
            XX, YY = np.meshgrid(xs_new, xs_new, indexing="ij")
            phi[k] = interp(np.stack([XX.ravel(), YY.ravel()], axis=-1)).reshape(target_N, target_N)

    phi = renormalize(np.clip(phi, 0.0, 1.0))
    return phi


# ── reference runner ─────────────────────────────────────────────────────────

