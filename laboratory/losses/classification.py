"""Classification objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def classification_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross entropy for ``(batch, classes)`` logits and integer labels."""
    if logits.dim() != 2:
        raise ValueError(f"expected (batch, classes) logits, got {tuple(logits.shape)}")
    return F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
