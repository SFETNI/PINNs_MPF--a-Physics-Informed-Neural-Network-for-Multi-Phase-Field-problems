"""GPU-native L-BFGS via TensorFlow Probability.

Replaces the legacy ``scipy.optimize.minimize(method='L-BFGS-B')`` outer loop,
which flattened the weights to NumPy and round-tripped CPU<->model every step.
Here the value-and-gradient closure runs entirely on the GPU in float64; only the
final converged position is read back.

Pattern: ``tf.dynamic_partition`` / ``tf.dynamic_stitch`` map a flat parameter
vector to/from the model's variable shapes inside a single ``tf.function``.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from ..config import DTYPE


def _function_factory(model, loss_fn: Callable[[], tf.Tensor]):
    trainables = model.trainable_variables
    shapes = [v.shape for v in trainables]
    n = len(shapes)

    idx, part, c = [], [], 0
    for i, shape in enumerate(shapes):
        cnt = int(np.prod(shape))
        idx.append(tf.reshape(tf.range(c, c + cnt, dtype=tf.int32), shape))
        part.extend([i] * cnt)
        c += cnt
    part = tf.constant(part, dtype=tf.int32)

    @tf.function
    def assign(flat):
        for i, p in enumerate(tf.dynamic_partition(flat, part, n)):
            trainables[i].assign(tf.reshape(p, shapes[i]))

    @tf.function
    def value_and_gradient(flat):
        assign(flat)
        with tf.GradientTape() as tape:
            loss = loss_fn()
        grads = tape.gradient(loss, trainables)
        grads = [g if g is not None else tf.zeros_like(v)
                 for g, v in zip(grads, trainables)]
        return loss, tf.dynamic_stitch(idx, grads)

    return value_and_gradient


def minimize(model, loss_fn, max_iterations: int = 5000, tolerance: float = 1e-12,
             f_relative_tolerance: float = 1e-12, num_correction_pairs: int = 50,
             max_line_search_iterations: int = 50, max_restarts: int = 5):
    """Run L-BFGS on ``model`` to minimize ``loss_fn()``; writes the result back.

    L-BFGS on PINNs can stall in a line-search failure well before convergence;
    restarting from the stuck position (fresh inverse-Hessian estimate) typically
    makes further progress. We restart until convergence, the iteration budget, or
    no further improvement.
    """
    vag = _function_factory(model, loss_fn)
    position = tf.cast(model.get_flat_params(), DTYPE)  # last KNOWN-GOOD (finite) position
    total_iters = 0
    last = None
    prev_obj = None
    for _ in range(max_restarts):
        res = tfp.optimizer.lbfgs_minimize(
            vag,
            initial_position=position,
            max_iterations=max_iterations,
            tolerance=tolerance,
            f_relative_tolerance=f_relative_tolerance,
            num_correction_pairs=num_correction_pairs,
            max_line_search_iterations=max_line_search_iterations,
        )
        obj = float(res.objective_value)
        finite = bool(tf.reduce_all(tf.math.is_finite(res.position))) and math.isfinite(obj)
        if not finite:
            break  # diverged to NaN/Inf: discard this step, keep the last good position
        position = res.position
        total_iters += int(res.num_iterations)
        last = res
        if bool(res.converged):
            break
        if prev_obj is not None and prev_obj - obj < 1e-15 * max(1.0, abs(prev_obj)):
            break  # restart made no progress
        prev_obj = obj
    model.set_flat_params(position)  # always a finite position
    if last is None:
        return SimpleResults(position=position, num_iterations=total_iters,
                             converged=False, failed=True, objective_value=float("inf"))
    return SimpleResults(position=position, num_iterations=total_iters,
                         converged=bool(last.converged), failed=bool(last.failed),
                         objective_value=float(last.objective_value))


class SimpleResults:
    """Aggregated result across restarts (mirrors the fields we use)."""

    def __init__(self, position, num_iterations, converged, failed, objective_value):
        self.position = position
        self.num_iterations = num_iterations
        self.converged = converged
        self.failed = failed
        self.objective_value = objective_value
