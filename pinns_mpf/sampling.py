"""Collocation / boundary point sampling (Latin Hypercube + domain edges)."""
from __future__ import annotations

import numpy as np
from pyDOE import lhs

from .config import NPDTYPE


def latin_hypercube(lb, ub, n: int, seed: int | None = None) -> np.ndarray:
    """n LHS samples in the box [lb, ub]. lb/ub are length-d arrays."""
    lb = np.asarray(lb, dtype=NPDTYPE)
    ub = np.asarray(ub, dtype=NPDTYPE)
    if seed is not None:
        np.random.seed(seed)
    return (lb + (ub - lb) * lhs(len(lb), n)).astype(NPDTYPE)


def edge_points(fixed_axis: int, fixed_value: float, free_samples: np.ndarray,
                d: int = 2) -> np.ndarray:
    """Points on a domain edge: fixed_axis held at fixed_value, other axis varies.

    free_samples: 1D array of coordinate values for the non-fixed axis.
    """
    free_samples = np.asarray(free_samples, dtype=NPDTYPE).reshape(-1)
    pts = np.empty((free_samples.size, d), dtype=NPDTYPE)
    pts[:, fixed_axis] = fixed_value
    pts[:, 1 - fixed_axis if d == 2 else (fixed_axis + 1) % d] = free_samples
    return pts
