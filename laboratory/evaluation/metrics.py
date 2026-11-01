"""Task-agnostic metrics.

Every function here takes model outputs (or ids) and returns a float, so it can
be reported by the training loop or used by an experiment's final evaluation.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> float:
    """Token-level accuracy over positions that are not ``ignore_index``."""
    mask = targets != ignore_index
    if not mask.any():
        return 0.0
    predictions = logits.argmax(dim=-1)
    return ((predictions == targets) & mask).sum().item() / mask.sum().item()


def perplexity(loss: float | torch.Tensor, ceiling: float = 20.0) -> float:
    """``exp(loss)``, clamped so an exploding early loss cannot overflow."""
    value = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
    return math.exp(min(value, ceiling))


def exact_match(
    predictions: Sequence[Sequence[int]],
    references: Sequence[Sequence[int]],
) -> float:
    """Fraction of predicted id sequences identical to the reference."""
    if len(predictions) != len(references):
        raise ValueError(
            f"got {len(predictions)} predictions and {len(references)} references"
        )
    if not predictions:
        return 0.0
    correct = sum(
        list(prediction) == list(reference)
        for prediction, reference in zip(predictions, references)
    )
    return correct / len(predictions)
