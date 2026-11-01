"""Metrics reported alongside the loss.

Metrics are pure functions of model outputs and targets, so they can be used
inside a ``step_fn`` and averaged by :class:`laboratory.training.Trainer`.
"""

from __future__ import annotations

import torch


class RunningMean:
    """Weighted running average, used for loss/metric reporting."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def mean(self) -> float:
        return self.total / max(self.count, 1)


def accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> float:
    """Token-level accuracy over positions that are not ``ignore_index``."""
    mask = targets != ignore_index
    if not mask.any():
        return 0.0
    predictions = logits.argmax(dim=-1)
    return ((predictions == targets) & mask).sum().item() / mask.sum().item()
