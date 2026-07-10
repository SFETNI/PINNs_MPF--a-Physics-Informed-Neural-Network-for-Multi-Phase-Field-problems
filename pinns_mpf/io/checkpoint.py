"""Weight checkpointing and metrics I/O — atomic + NaN-safe (no pickle).

Writes go to a temp file then os.replace() (atomic on POSIX), so a SIGTERM/SIGKILL
mid-write under the run time-cap can never leave a corrupt checkpoint. JSON sanitizes
NaN/Inf to null so metrics.json is always valid (the review found bare NaN breaks it).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np


def _atomic(path: str, write_fn) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    write_fn(tmp)
    os.replace(tmp, path)


def save_weights(model, path: str) -> None:
    arrays = [v.numpy() for v in model.trainable_variables]
    target = path if path.endswith(".npz") else path + ".npz"
    # np.savez appends .npz to a non-.npz name; write to an explicit .npz tmp then replace.
    tmp = target + ".tmp.npz"
    np.savez(tmp, *arrays)
    os.makedirs(os.path.dirname(os.path.abspath(target)) or ".", exist_ok=True)
    os.replace(tmp, target)


def load_weights(model, path: str) -> None:
    data = np.load(path if path.endswith(".npz") else path + ".npz")
    flat = np.concatenate([data[k].ravel() for k in data.files])
    model.set_flat_params(flat)


def _sanitize(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    return o


def save_json(obj: dict, path: str) -> None:
    payload = json.dumps(_sanitize(obj), indent=2, default=float, allow_nan=False)
    _atomic(path, lambda p: open(p, "w").write(payload))
