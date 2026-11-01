"""Plot helpers used by comparison experiments.

Matplotlib is imported lazily so importing :mod:`laboratory` never requires it.
Every function writes a PNG and returns its path; experiments keep the data in
JSON, so plots can be regenerated without retraining.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_lines(
    series: Mapping[str, tuple[Sequence[float], Sequence[float], Sequence[float] | None]],
    path: str | Path,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    logy: bool = False,
) -> Path:
    """Plot named ``(x, y, yerr)`` series; ``yerr=None`` draws a plain line."""
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    for name, (x, y, error) in series.items():
        axis.plot(x, y, marker="o", markersize=3, linewidth=1.6, label=name)
        if error is not None:
            axis.fill_between(x, np.asarray(y) - np.asarray(error), np.asarray(y) + np.asarray(error), alpha=0.15)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    if logy:
        axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    path = Path(path)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_bars(
    values: Mapping[str, tuple[float, float]],
    path: str | Path,
    *,
    ylabel: str,
    title: str,
) -> Path:
    """Bar chart of ``name -> (mean, error)`` with error bars."""
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    names = list(values)
    means = [values[name][0] for name in names]
    errors = [values[name][1] for name in names]
    axis.bar(names, means, yerr=errors, capsize=4, color=["#4c72b0", "#dd8452", "#55a868"])
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = Path(path)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_gradient_decay(
    curves: Mapping[str, Sequence[float]],
    path: str | Path,
    *,
    title: str = "Gradient reach of the final hidden state",
) -> Path:
    """Log-scale norm of dL/dh_t against distance from the final timestep."""
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    for name, values in curves.items():
        lags = np.arange(len(values) - 1, -1, -1)
        axis.semilogy(lags, np.asarray(values, dtype=float), linewidth=1.6, label=name)
    axis.set_xlabel("lag from the final timestep")
    axis.set_ylabel("mean ||dL/dh_t||  (log scale)")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    path = Path(path)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
