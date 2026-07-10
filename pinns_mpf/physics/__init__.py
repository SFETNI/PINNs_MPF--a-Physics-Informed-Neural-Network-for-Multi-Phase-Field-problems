"""pinns_mpf.physics — PDE residuals (dimensionless MPF)."""
from .single_phase import dh_dphi, double_well, pde_residual

__all__ = ["dh_dphi", "double_well", "pde_residual"]
