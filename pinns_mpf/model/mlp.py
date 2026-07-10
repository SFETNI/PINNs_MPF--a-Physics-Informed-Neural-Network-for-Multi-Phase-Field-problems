"""Fully-connected PINN network (float64, GPU-resident).

Input handling per raw dimension:
- non-periodic dims  -> min-max normalized to [0,1];
- periodic dims      -> [sin, cos](2*pi*(x-lb)/L) Fourier embedding, so the network is
  exactly periodic with period L=ub-lb (used for coarsening benchmarks whose references
  use periodic BCs -> removes the need for explicit boundary losses). The embedding is
  smooth, so spatial derivatives in the PDE residual flow through it (chain rule).

tanh hidden activations; linear final layer + optional output activation
(``sigmoid`` for a single bounded phase field; ``linear``/None for multi-phase logits
fed to a softmax). Xavier-normal init. Flat get/set for TFP L-BFGS.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import tensorflow as tf

from ..config import DTYPE

_TWO_PI = tf.constant(2.0 * np.pi, dtype=DTYPE)


class MLP(tf.Module):
    def __init__(
        self,
        layers: Sequence[int],
        lb,
        ub,
        output_activation: str = "sigmoid",
        seed: int = 1234,
        periodic_dims: Sequence[int] | None = None,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.layers = list(layers)
        # Place constants and weight variables on GPU so matmuls run on GPU in eager mode.
        # tf.Variable colocation: ops that read a variable run on the variable's device.
        # Without explicit GPU placement, variables default to CPU and matmuls stay on CPU.
        with tf.device('/GPU:0'):
            self.lb = tf.constant(np.asarray(lb, dtype=np.float64).reshape(1, -1), dtype=DTYPE)
            self.ub = tf.constant(np.asarray(ub, dtype=np.float64).reshape(1, -1), dtype=DTYPE)
        self.output_activation = output_activation
        self.in_dim = self.layers[0]
        self.periodic_set = set(periodic_dims or [])
        # feature dim after embedding: periodic dims -> 2 (sin,cos), others -> 1
        self.feat_dim = sum(2 if j in self.periodic_set else 1 for j in range(self.in_dim))

        self.W: list[tf.Variable] = []  # [w0,b0,w1,b1,...] (tf.Module tracks the list)
        self.num_parameters = 0
        init = tf.random.Generator.from_seed(seed)
        with tf.device('/GPU:0'):
            for i in range(len(self.layers) - 1):
                din = self.feat_dim if i == 0 else self.layers[i]
                dout = self.layers[i + 1]
                std = np.sqrt(2.0 / (din + dout))  # Xavier
                w = tf.Variable(init.normal([din, dout], dtype=DTYPE) * std,
                                trainable=True, name=f"w{i + 1}")
                b = tf.Variable(tf.zeros([dout], dtype=DTYPE), trainable=True, name=f"b{i + 1}")
                self.W += [w, b]
                self.num_parameters += din * dout + dout

    def _features(self, X):
        cols = []
        for j in range(self.in_dim):
            xj = X[:, j:j + 1]
            lo = self.lb[0, j]
            span = self.ub[0, j] - lo
            if j in self.periodic_set:
                arg = _TWO_PI * (xj - lo) / span
                cols += [tf.sin(arg), tf.cos(arg)]
            else:
                cols.append((xj - lo) / span)
        return tf.concat(cols, axis=1)

    @tf.function(reduce_retracing=True)
    def __call__(self, X):
        H = self._features(X)
        for i in range(len(self.layers) - 2):
            H = tf.nn.tanh(tf.matmul(H, self.W[2 * i]) + self.W[2 * i + 1])
        Y = tf.matmul(H, self.W[-2]) + self.W[-1]
        if self.output_activation == "sigmoid":
            Y = tf.nn.sigmoid(Y)
        elif self.output_activation not in (None, "linear"):
            raise ValueError(f"unknown output_activation: {self.output_activation}")
        return Y

    # ---- flat parameter views (for L-BFGS) ----
    def get_flat_params(self) -> tf.Tensor:
        return tf.concat([tf.reshape(v, [-1]) for v in self.trainable_variables], axis=0)

    def set_flat_params(self, flat) -> None:
        flat = tf.convert_to_tensor(flat, dtype=DTYPE)
        i = 0
        for v in self.trainable_variables:
            n = int(np.prod(v.shape))
            v.assign(tf.reshape(flat[i:i + n], v.shape))
            i += n
