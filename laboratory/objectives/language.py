"""Objectives for token prediction.

The same function drives decoder-only language models and encoder-decoder
sequence models: both produce ``(..., vocabulary)`` logits and integer targets,
and both need padding positions ignored.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def token_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross entropy over ``(..., vocabulary)`` logits.

    Leading dimensions are flattened, so ``(batch, sequence, vocabulary)`` and
    ``(batch, vocabulary)`` both work. Targets equal to ``ignore_index``
    (padding) contribute no loss and no gradient.
    """
    if logits.shape[:-1] != targets.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} implies targets of shape "
            f"{tuple(logits.shape[:-1])}, got {tuple(targets.shape)}"
        )
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )
