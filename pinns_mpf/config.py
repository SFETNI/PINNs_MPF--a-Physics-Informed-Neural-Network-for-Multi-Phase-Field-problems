"""Runtime configuration: precision, device, seeds, and physical parameters.

Single source of truth for dtype (``float64``) and GPU setup. The original code
pinned everything to ``/CPU:0`` and used ``float32`` casts ad hoc; here we keep
``float64`` everywhere and let TensorFlow place ops on the GPU.
"""
from __future__ import annotations

import dataclasses
import os

import numpy as np
import tensorflow as tf

# High precision is the project goal; float64 end-to-end.
DTYPE = tf.float64
NPDTYPE = np.float64


def setup(seed: int = 1234, memory_growth: bool = True, log_level: str = "2") -> dict:
    """Configure TF for reproducible float64 GPU runs. Returns a device-info dict."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", log_level)  # (legacy had a typo here)
    tf.keras.backend.set_floatx("float64")

    gpus = tf.config.list_physical_devices("GPU")
    if memory_growth:
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except RuntimeError:
                pass  # already initialized

    np.random.seed(seed)
    tf.random.set_seed(seed)

    return {
        "tf_version": tf.__version__,
        "gpus": [g.name for g in gpus],
        "dtype": "float64",
        "seed": seed,
    }


@dataclasses.dataclass(frozen=True)
class PhysicalParams:
    """Dimensionless MPF parameters (paper Eqs. 17-24).

    For the single-phase traveling wave the analytic interface speed is
    ``v_n = mu * delta_g`` (curvature term vanishes for the equilibrium profile).
    """

    mu: float = 1.0e-6      # interface mobility
    sigma: float = 1.0      # interface energy
    eta: float = 1.0        # interface width
    delta_g: float = 5.0e5  # constant driving force

    @property
    def v_n(self) -> float:
        return self.mu * self.delta_g
