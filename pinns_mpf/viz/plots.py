"""Validation plots (non-interactive Agg backend)."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _savefig(fig, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_solution_vs_reference(phi_pred, phi_true, x, t, path, n_slices=4):
    """phi_pred/phi_true: [nx, nt]. Heatmaps (pred, ref, |error|) + time slices."""
    err = np.abs(phi_pred - phi_true)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    extent = [t.min(), t.max(), x.min(), x.max()]
    for ax, data, title in zip(
        axes[0], [phi_pred, phi_true, err],
        [r"$\phi_{pred}$", r"$\phi_{ref}$", r"$|\phi_{pred}-\phi_{ref}|$"],
    ):
        im = ax.imshow(data, origin="lower", aspect="auto", extent=extent, cmap="rainbow")
        ax.set_xlabel("t"); ax.set_ylabel("x"); ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)

    for ax in axes[1]:
        ax.remove()
    ax = fig.add_subplot(2, 1, 2)
    idxs = np.linspace(0, len(t) - 1, n_slices).astype(int)
    colors = plt.cm.viridis(np.linspace(0, 1, n_slices))
    for j, c in zip(idxs, colors):
        ax.plot(x, phi_true[:, j], "-", color=c, lw=2)
        ax.plot(x, phi_pred[:, j], "--", color=c, lw=1.5)
        ax.text(0.0, 0.0, "", color=c)
    ax.plot([], [], "k-", label="reference"); ax.plot([], [], "k--", label="PINN")
    ax.set_xlabel("x"); ax.set_ylabel(r"$\phi$"); ax.legend()
    ax.set_title("time slices (solid=reference, dashed=PINN)")
    _savefig(fig, path)


def plot_loss(history, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for key in ("total", "pde", "ic", "bc"):
        ys = [h[key] for h in history if key in h]
        if ys:
            ax.semilogy(range(len(ys)), ys, label=key, marker=".")
    ax.set_xlabel("log point"); ax.set_ylabel("loss"); ax.legend()
    ax.set_title("training losses")
    _savefig(fig, path)
