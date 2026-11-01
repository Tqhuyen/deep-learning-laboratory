"""Checkpoint saving and loading.

A checkpoint holds model weights, optional optimizer state and arbitrary
metadata (config, step, seed). Loading returns the metadata so an experiment can
restore its configuration instead of guessing it.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    **extra,
) -> None:
    """Save weights plus optional optimizer state and metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), **extra}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load what :func:`save_checkpoint` wrote; returns the metadata."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return {k: v for k, v in payload.items() if k not in ("model", "optimizer")}
