"""Modernized PINNs-MPF implementation.

This package contains the float64 PINN/MPF source used by the current local
validation package: analytic single-network baselines, single-phase
MultiNN/time-marched grain shrinkage, and multiphase MultiNN triple-junction
training with softmax phase coupling.
"""
from . import config
from .config import DTYPE, PhysicalParams, setup
from .model.mlp import MLP
from .training import SinglePhaseTrainer

__version__ = "2.0.0"
__all__ = ["config", "setup", "PhysicalParams", "DTYPE", "MLP", "SinglePhaseTrainer"]
