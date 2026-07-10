"""pinns_mpf.losses — PINN loss terms."""
from .single_phase import loss_bc_dirichlet, loss_ic, loss_pde, total_loss

__all__ = ["loss_bc_dirichlet", "loss_ic", "loss_pde", "total_loss"]
