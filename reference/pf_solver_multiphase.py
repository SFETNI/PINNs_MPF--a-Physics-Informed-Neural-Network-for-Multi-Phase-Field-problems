"""Multi-phase finite-difference phase-field reference solver (NumPy, CPU-only).

Ground truth for Benchmark 5 (triple junction). Ports the legacy B5 multi-phase
MPF model (paper Eqs. 22-23, equal interfacial energies, no driving force):

    I_alpha          = lap(phi_alpha) + (pi^2 / eta^2) phi_alpha
    dphi_alpha/dt    = (mu/N) * sum_{beta != alpha} sigma * (I_alpha - I_beta)
                     = mu sigma (I_alpha - mean_beta I_beta)

The mean form makes sum_alpha dphi_alpha/dt = 0, so the constraint sum_alpha phi=1
is conserved by construction (we still renormalize for round-off). For N=2 this
reduces *exactly* to the single-phase curvature equation in pf_solver, which is the
rigorous cross-check below.

NumPy only -> zero GPU footprint; runs on a GPU-less box.
"""
from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class MPParams:
    mu: float = 1.0
    sigma: float = 1.0
    eta: float = 0.05


def laplacian_stack(phis, dx):
    """5-point periodic Laplacian over a [N, Ny, Nx] stack."""
    return (np.roll(phis, 1, -2) + np.roll(phis, -1, -2)
            + np.roll(phis, 1, -1) + np.roll(phis, -1, -1) - 4.0 * phis) / dx ** 2


def dphi_dt(phis, p: MPParams, dx):
    I = laplacian_stack(phis, dx) + (np.pi ** 2 / p.eta ** 2) * phis
    return p.mu * p.sigma * (I - I.mean(axis=0, keepdims=True))


def stable_dt(p: MPParams, dx, safety=0.2):
    diff = dx ** 2 / (4.0 * p.mu * p.sigma)
    react = 1.0 / (p.mu * p.sigma * np.pi ** 2 / p.eta ** 2)
    return safety * min(diff, react)


def renormalize(phis):
    phis = np.clip(phis, 0.0, 1.0)
    s = phis.sum(axis=0, keepdims=True)
    return phis / np.where(s > 1e-12, s, 1.0)


def simulate(phis0, p: MPParams, dx, dt=None, t_final=None, nsteps=None, save_every=1):
    dt = dt or stable_dt(p, dx)
    if nsteps is None:
        nsteps = int(np.ceil(t_final / dt))
    phis = renormalize(phis0.copy())
    times, frames, sum_err = [0.0], [phis.copy()], [float(np.max(np.abs(phis.sum(0) - 1)))]
    for k in range(1, nsteps + 1):
        phis = renormalize(phis + dt * dphi_dt(phis, p, dx))
        if k % save_every == 0 or k == nsteps:
            times.append(k * dt)
            frames.append(phis.copy())
            sum_err.append(float(np.max(np.abs(phis.sum(0) - 1))))
    return np.array(times), frames, np.array(sum_err)


# ---------------- initial conditions ----------------
def _grid(n, L):
    xs = np.linspace(0.0, L, n, endpoint=False)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    return xs, X, Y


def init_circle_2phase(n, L, R0, eta, center=None):
    """Phase 0 = circular grain (radial sin profile), phase 1 = matrix. sum=1."""
    cx, cy = center or (L / 2, L / 2)
    _, X, Y = _grid(n, L)
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    phi0 = np.where(r < R0 - eta / 2, 1.0,
                    np.where(r > R0 + eta / 2, 0.0,
                             0.5 - 0.5 * np.sin(np.pi * (r - R0) / eta)))
    return np.stack([phi0, 1.0 - phi0]).astype(np.float64)


def init_t_junction(n, L=1.0, num_phases=4):
    """Non-equilibrium grain partition that relaxes toward Young's 120 deg (equal sigma).
    num_phases=3: T-junction (top / bottom-left / bottom-right), initial 90/90/180.
    num_phases=4: double-T (one top grain over three bottom grains) -> two triple junctions,
    matching the PINN-Phase paper's 4-phase triple-junction benchmark (Table 2/6).
    Works in any units (uses L/2, L/3 — not hardcoded 0.5)."""
    _, X, Y = _grid(n, L)
    h = L / 2.0
    if num_phases == 3:
        regions = [Y >= h, (Y < h) & (X < h), (Y < h) & (X >= h)]
    elif num_phases == 4:
        a, b = L / 3.0, 2.0 * L / 3.0
        regions = [Y >= h,
                   (Y < h) & (X < a),
                   (Y < h) & (X >= a) & (X < b),
                   (Y < h) & (X >= b)]
    else:
        raise ValueError("num_phases must be 3 or 4")
    return np.stack([r.astype(np.float64) for r in regions])


def init_90deg_periodic(n, L=1.0, num_phases=4):
    """Fully periodic-safe orthogonal IC with 6 *interior* triple junctions.

    Same orthogonal 90/90/180 junction geometry as the paper's B5, arranged so
    that NEITHER periodic seam carries a phase interface:

      * x-seam periodic-safe: Phase 2 wraps both x-edges (x < L/6 OR x >= 5L/6).
      * y-seam periodic-safe: Phase 1 is a centered horizontal band [L/4, 3L/4); the
        complement (top+bottom strips) wraps the y=0/L seam continuously.

    Both horizontal interfaces (y=L/4 and y=3L/4) are therefore interior, giving
    6 interior triple junctions at

        (L/6, L/4) (L/2, L/4) (5L/6, L/4)   and   (L/6, 3L/4) (L/2, 3L/4) (5L/6, 3L/4)

    all at x in [0.167L, 0.833L], y in {0.25L, 0.75L} — well outside the angle
    detector's margin band (margin_frac<=0.12), so junction_angles_all detects a
    stable count of 6 across the whole trajectory.

    If the horizontal body is split directly across the periodic y-seam, the seam
    itself becomes a physical interface and produces spurious seam junctions. The
    centered-band layout keeps both horizontal interfaces inside the domain, giving
    six interior junctions while preserving the orthogonal 90/90/180 geometry.

    Initial angles: 90/90/180 at each junction; RMS ~= 42.4 deg from 120 deg.
    Interfaces start sharp (binary); diffuse them with a few FD burn-in steps
    before the main simulation (see generate_b5_article_reference.run_case).
    """
    if num_phases != 4:
        raise ValueError("init_90deg_periodic supports num_phases=4 only")
    _, X, Y = _grid(n, L)
    yq, yb = L / 4.0, 3.0 * L / 4.0        # centered band edges (interior interfaces)
    a, b, h = L / 6.0, 5.0 * L / 6.0, L / 2.0
    mid = (Y >= yq) & (Y < yb)             # Phase 1: centered horizontal band
    comp = ~mid                            # complement: top+bottom, wraps y-seam
    regions = [
        mid,
        comp & ((X < a) | (X >= b)),       # Phase 2: both periodic x-edges
        comp & (X >= a) & (X < h),         # Phase 3: left-center segment
        comp & (X >= h) & (X < b),         # Phase 4: right-center segment
    ]
    return np.stack([r.astype(np.float64) for r in regions])


def init_triple_junction_direct(n, L=1.0):
    """Four-phase triple-junction IC with equal phase areas.

    Layout:
      Phase 0 (center-top):    x ∈ [L/4, 3L/4), y ∈ [L/2, L)    — 25% area
      Phase 1 (center-bottom): x ∈ [L/4, 3L/4), y ∈ [0,  L/2)   — 25% area
      Phase 2 (side-middle):   x ∉ center,       y ∈ [L/4, 3L/4) — 25% area
      Phase 3 (corners):       remainder                           — 25% area

    Six interior T-junctions at (L/4, L/4), (3L/4, L/4), (L/4, L/2), (3L/4, L/2),
    (L/4, 3L/4), (3L/4, 3L/4). Initial local geometry: 90/90/180 at each.

    Periodic seam note: at the y-seam (y=0/L) the center column transitions from
    Phase 1 (bottom) to Phase 0 (top), creating 2 extra physical seam interfaces.
    The junction detector's margin band (margin_frac=0.10, N=128 → 13-cell margin)
    masks these; interior 6-junction count is unaffected. See
    init_triple_junction_periodic_safe for a strictly seam-safe variant.
    """
    _, X, Y = _grid(n, L)
    a, b = L / 4.0, 3.0 * L / 4.0
    ym = L / 2.0
    yL, yH = L / 4.0, 3.0 * L / 4.0
    center = (X >= a) & (X < b)
    side = ~center
    ph0 = center & (Y >= ym)
    ph1 = center & (Y < ym)
    ph2 = side & (Y >= yL) & (Y < yH)
    ph3 = ~ph0 & ~ph1 & ~ph2
    return np.stack([ph0, ph1, ph2, ph3]).astype(np.float64)


def init_triple_junction_periodic_safe(n, L=1.0, y_cap_frac=1.0 / 16):
    """Periodic-safe four-phase triple-junction IC.

    Identical topology to init_triple_junction_direct but a cap of height
    cap = y_cap_frac × L
    is cut from the top and bottom of both center grains (Phases 0 and 1) and
    replaced with Phase 3 (corners). This makes the y-seam pass entirely through
    Phase 3 in the center column.

    With y_cap_frac = 1/16 (8 pixels at N=128):
      Phase 0 ≈ 21.9%  Phase 1 ≈ 21.9%  Phase 2 = 25%  Phase 3 ≈ 31.3%

    Six interior junctions at the same positions as init_triple_junction_direct.
    The cap-boundary lines (y = cap and y = L-cap, x ∈ center) are binary
    Ph0/Ph3 and Ph1/Ph3 interfaces, not triple junctions.
    """
    _, X, Y = _grid(n, L)
    a, b = L / 4.0, 3.0 * L / 4.0
    ym = L / 2.0
    yL, yH = L / 4.0, 3.0 * L / 4.0
    cap = y_cap_frac * L
    center = (X >= a) & (X < b)
    side = ~center
    ph0 = center & (Y >= ym) & (Y < L - cap)
    ph1 = center & (Y >= cap) & (Y < ym)
    ph2 = side & (Y >= yL) & (Y < yH)
    ph3 = ~ph0 & ~ph1 & ~ph2
    return np.stack([ph0, ph1, ph2, ph3]).astype(np.float64)


def init_90deg_equal_area(n, L=1.0):
    """Article-style equal-area 90-degree IC with 6 interior triple junctions.

    All four phases have exactly 25% area:
      Phase 0: horizontal connector band y ∈ [3L/8, 5L/8), area = L/4 × L = 25%.
      Phase 1: complement and (x < L/6 or x >= 5L/6), edge-wrapped, area 25%.
      Phase 2: complement and L/6 <= x < L/2, area 25%.
      Phase 3: complement and L/2 <= x < 5L/6, area 25%.

    Six interior T-junctions at:
      (L/6, 3L/8), (L/2, 3L/8), (5L/6, 3L/8)
      (L/6, 5L/8), (L/2, 5L/8), (5L/6, 5L/8)

    Periodic-seam behavior:
      x-seam (x=0/L): Phase 1 wraps both edges — periodic-safe.
      y-seam (y=0/L): y=0 is in the complement (Phase 0 is at [3L/8, 5L/8)) — periodic-safe.

    Initial geometry: 90/90/180 at each junction.
    Interfaces start sharp (binary); use burn-in steps before the main simulation.
    """
    _, X, Y = _grid(n, L)
    yq = 3.0 * L / 8.0   # band lower edge
    yb = 5.0 * L / 8.0   # band upper edge
    a = L / 6.0           # x-boundary 1 (between Phase 1 and Phase 2)
    h = L / 2.0           # x-boundary 2 (between Phase 2 and Phase 3)
    b = 5.0 * L / 6.0     # x-boundary 3 (between Phase 3 and Phase 1)
    band = (Y >= yq) & (Y < yb)   # Phase 0: narrow centered band
    comp = ~band                   # complement: top + bottom strips
    regions = [
        band,
        comp & ((X < a) | (X >= b)),   # Phase 1: both periodic x-edges
        comp & (X >= a) & (X < h),     # Phase 2: left-center
        comp & (X >= h) & (X < b),     # Phase 3: right-center
    ]
    return np.stack([r.astype(np.float64) for r in regions])


def triple_point(phis, L=1.0, margin_frac=0.12):
    """Interior cell where the three largest phases best coexist (product of the
    top-3 sorted phase values). A boundary margin excludes spurious corner/edge maxima."""
    n = phis.shape[-1]
    s = np.sort(phis, axis=0)                       # ascending; top-3 = last three
    score = s[-1] * s[-2] * s[-3] if phis.shape[0] >= 3 else np.prod(phis, axis=0)
    m = max(1, int(margin_frac * n))
    masked = np.full_like(score, -1.0)
    masked[m:-m, m:-m] = score[m:-m, m:-m]
    i, j = np.unravel_index(np.argmax(masked), masked.shape)
    return i, j, (i + 0.5) * L / n, (j + 0.5) * L / n


def junction_angles(phis, L=1.0, radii_frac=(0.04, 0.06, 0.08, 0.10)):
    """Three sector angles (deg) at the triple point; ~120/120/120 at equilibrium.
    Scans several ring radii and accepts the first that robustly shows exactly three
    distinct dominant phases (robust to thin interfaces / off-center junctions).
    Returns None if no radius gives a robust 3-phase ring."""
    n = phis.shape[-1]
    i0, j0, _, _ = triple_point(phis, L)
    thetas = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    for rf in radii_frac:
        rad = max(2, int(rf * n))
        ii = (np.rint(i0 + rad * np.cos(thetas)).astype(int)) % n
        jj = (np.rint(j0 + rad * np.sin(thetas)).astype(int)) % n
        dom = np.argmax(phis[:, ii, jj], axis=0)    # dominant phase around the ring
        ch = np.where(dom != np.roll(dom, 1))[0]
        if len(set(dom.tolist())) == 3 and len(ch) == 3:
            bnd = np.sort(thetas[ch])
            gaps = np.diff(np.concatenate([bnd, [bnd[0] + 2 * np.pi]]))
            return np.degrees(gaps)
    return None
