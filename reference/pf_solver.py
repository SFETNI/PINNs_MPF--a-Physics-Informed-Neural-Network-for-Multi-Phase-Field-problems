"""Finite-difference phase-field reference solver (NumPy, CPU-only).

Ground-truth generator for Benchmarks 2-5. Solves the same dimensionless
single-phase MPF equation as ``pinns_mpf.physics.single_phase`` (paper Eq. 17),
with an explicit Euler step, a 5-point Laplacian, and periodic boundaries — a
direct port of the legacy ``Phase_Field_resolution`` solver:

    phi_t = mu [ sigma ( lap phi + (pi^2/2 eta^2)(2 phi - 1) )
                 + (pi/eta) sqrt(phi(1-phi)) * delta_g ]

For a circular grain with no driving force this reproduces the curvature law
R(t)^2 = R0^2 - 2 mu sigma t (used as the validation check below).

NOTE: imports NumPy only — never TensorFlow — so it has zero GPU footprint and
can run while the GPU is busy.
"""
from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class PFParams:
    mu: float = 1.0
    sigma: float = 1.0
    eta: float = 0.05
    delta_g: float = 0.0


def laplacian_periodic(phi, dx):
    return (np.roll(phi, 1, 0) + np.roll(phi, -1, 0)
            + np.roll(phi, 1, 1) + np.roll(phi, -1, 1) - 4.0 * phi) / dx ** 2


def dphi_dt(phi, p: PFParams, dx):
    lap = laplacian_periodic(phi, dx)
    well = np.pi ** 2 / (2.0 * p.eta ** 2) * (2.0 * phi - 1.0)
    dh = np.pi / p.eta * np.sqrt(np.clip(phi * (1.0 - phi), 0.0, None))
    return p.mu * (p.sigma * (lap + well) + dh * p.delta_g)


def projected_dphi_dt(phi, p: PFParams, dx, eps=1e-3):
    """Bound-aware EFFECTIVE RHS that matches the clipped explicit integrator.

    The raw Allen-Cahn RHS pushes phi<0 in the phi=0 matrix and phi>1 in the
    phi=1 core (the local well term (pi^2/2eta^2)(2phi-1) is outward-pointing in
    the saturated bulk). The reference ``simulate`` clips phi to [0,1] each step,
    so the *effective* dynamics at the bounds are zero there, not the raw RHS:

        if phi <= eps   and r < 0:  -> 0   (would leave the [0,1] box downward)
        if phi >= 1-eps and r > 0:  -> 0   (would leave the [0,1] box upward)
        otherwise:                  -> r

    Training the PINN against this projected RHS (instead of the raw RHS) removes
    the artificial bulk-grow residual, so denoising is no longer the mechanism
    that holds the bulk -- it becomes a light safety control only.
    """
    r = dphi_dt(phi, p, dx)
    out_low = (phi <= eps) & (r < 0.0)
    out_high = (phi >= 1.0 - eps) & (r > 0.0)
    return np.where(out_low | out_high, 0.0, r)


def projection_diagnostics(phi, p: PFParams, dx, eps=1e-3, band=(0.02, 0.98)):
    """Where/how much the projected RHS differs from the raw RHS, split by region.

    Returns fractions and projected-away magnitudes over three regions:
    bulk-low (phi<band[0]), interface band, bulk-high (phi>band[1]). If projection
    only fires in the bulk (expected) the interface fraction should be ~0; a large
    interface fraction would mean the bound projection is corrupting the moving
    front (a failure mode to flag)."""
    r = dphi_dt(phi, p, dx)
    rp = projected_dphi_dt(phi, p, dx, eps=eps)
    diff = np.abs(r - rp)
    fired = diff > 0.0
    lo = phi < band[0]
    hi = phi > band[1]
    mid = ~lo & ~hi
    out = {"eps": float(eps),
           "frac_total": float(np.mean(fired)),
           "max_projected_away": float(diff.max()) if diff.size else 0.0,
           "mean_projected_away": float(diff[fired].mean()) if fired.any() else 0.0}
    for name, mask in (("bulk_low", lo), ("interface", mid), ("bulk_high", hi)):
        n = int(mask.sum())
        out[f"frac_{name}"] = float(fired[mask].mean()) if n else 0.0
    return out


def free_energy_components(phi, p: PFParams, dx):
    """Split the Allen-Cahn free energy into its GRADIENT and DOUBLE-WELL parts:
        F_grad = sigma * sum[ 0.5|grad phi|^2 ] dx^2          (derivative term)
        F_well = sigma * (pi^2/2eta^2) sum[ phi(1-phi) ] dx^2 (double-well / interfacial amount)
    Returns (F_total, F_grad, F_well).

    Why the split matters for a PINN: F_grad amplifies high-frequency representation noise
    (a slightly rough network field has large extra |grad phi|^2), so F_total can RISE across
    moving-IC handoffs even when the physics (area/radius) is correct. F_well is derivative-free
    and ~ interface length ~ R, so it is a ROBUST monotone shrinkage signal; F_grad doubles as a
    field-roughness early-warning. (Delta_g bulk term omitted -- validated benchmark is Delta_g=0.)"""
    gx = (np.roll(phi, -1, 0) - np.roll(phi, 1, 0)) / (2.0 * dx)
    gy = (np.roll(phi, -1, 1) - np.roll(phi, 1, 1)) / (2.0 * dx)
    F_grad = float(p.sigma * np.sum(0.5 * (gx ** 2 + gy ** 2)) * dx ** 2)
    F_well = float(p.sigma * (np.pi ** 2 / (2.0 * p.eta ** 2))
                   * np.sum(np.clip(phi * (1.0 - phi), 0.0, None)) * dx ** 2)
    return F_grad + F_well, F_grad, F_well


def free_energy(phi, p: PFParams, dx):
    """Total Allen-Cahn interfacial free energy (gradient + double-well). Its variational
    derivative -dF/dphi = sigma(lap phi + (pi^2/2eta^2)(2phi-1)) is exactly the Delta_g=0 RHS,
    so curvature-driven shrinkage is gradient flow of F. See free_energy_components for the split."""
    return free_energy_components(phi, p, dx)[0]


def diagnose_frame(phi, p: PFParams, dx, eps=1e-3):
    """Per-frame scalar diagnostics used by reference verification and PINN validation."""
    phic = np.clip(phi, 0.0, 1.0)
    interface = (phic > 0.02) & (phic < 0.98)
    return {"radius": float(grain_radius(phic, dx)),
            "mass": float(np.sum(phic) * dx ** 2),
            "max_phi": float(phic.max()),
            "free_energy": free_energy(phic, p, dx),
            "interface_present": bool(np.any(interface)),
            "projection": projection_diagnostics(phic, p, dx, eps=eps)}


def stable_dt(p: PFParams, dx, safety=0.2):
    """CFL-style step for curvature, reaction, and driving-force motion.

    For delta_g=0 this reduces to the previous diffusion/reaction limit. With a
    strong driving force, the interface-motion limit keeps the front displacement
    below about dx/2 per explicit step.
    """
    diff = dx ** 2 / (4.0 * p.mu * p.sigma)
    react = 1.0 / (p.mu * p.sigma * np.pi ** 2 / p.eta ** 2)
    if abs(p.delta_g) > 0.0:
        drive_cfl = (dx / 2.0) / (p.mu * abs(p.delta_g))
        return min(safety * min(diff, react), drive_cfl)
    return safety * min(diff, react)


def init_circle(n, L, R0, eta, center=None, periodic=True):
    """Radial sin interface profile: phi=1 inside the grain, 0 outside.

    ``periodic`` uses the minimum-image distance so the IC is consistent with the
    periodic Laplacian (identical to the direct distance for a centered grain that
    does not reach the boundary; matters only if the grain straddles a wrap edge)."""
    center = center if center is not None else (L / 2.0, L / 2.0)
    xs = np.linspace(0.0, L, n, endpoint=False)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    dX = X - center[0]
    dY = Y - center[1]
    if periodic:
        dX -= L * np.round(dX / L)
        dY -= L * np.round(dY / L)
    r = np.sqrt(dX ** 2 + dY ** 2)
    phi = np.where(r < R0 - eta / 2, 1.0,
                   np.where(r > R0 + eta / 2, 0.0,
                            0.5 - 0.5 * np.sin(np.pi * (r - R0) / eta)))
    return phi.astype(np.float64), xs


def grain_radius(phi, dx):
    """Effective radius from the phi=1 area: R = sqrt(area / pi)."""
    area = np.sum(phi) * dx ** 2
    return np.sqrt(area / np.pi)


def simulate(phi0, p: PFParams, dx, dt=None, t_final=None, nsteps=None, save_every=1):
    """Evolve phi0. Returns (times, frames[list of phi], radii). Clips phi to [0,1]."""
    dt = dt or stable_dt(p, dx)
    if nsteps is None:
        nsteps = int(np.ceil(t_final / dt))
    phi = phi0.copy()
    times, frames, radii = [0.0], [phi.copy()], [grain_radius(phi, dx)]
    for k in range(1, nsteps + 1):
        phi = np.clip(phi + dt * dphi_dt(phi, p, dx), 0.0, 1.0)
        if k % save_every == 0 or k == nsteps:
            times.append(k * dt)
            frames.append(phi.copy())
            radii.append(grain_radius(phi, dx))
    return np.array(times), frames, np.array(radii)


def curvature_law_radius(t, R0, mu, sigma):
    """Analytic shrinkage (no driving force): R(t)^2 = R0^2 - 2 mu sigma t."""
    return np.sqrt(np.clip(R0 ** 2 - 2.0 * mu * sigma * t, 0.0, None))
