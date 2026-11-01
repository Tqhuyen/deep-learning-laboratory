"""Learning-rate schedules.

All schedules are ``torch.optim.lr_scheduler.LambdaLR`` objects and are stepped
once per optimizer step, which is what :class:`laboratory.training.Trainer`
does. If a paper defines a different shape, add a function here that returns a
``LambdaLR`` with the same contract.
"""

from __future__ import annotations

import math

import torch


def cosine_warmup(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
    """Linear warmup, then cosine decay to ``min_lr_ratio * base_lr``."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def noam(optimizer, d_model: int, warmup_steps: int):
    """The schedule from Vaswani et al. (2017): ``d^-0.5 * min(step^-0.5, step*warmup^-1.5)``."""

    def factor(step: int) -> float:
        step = max(step, 1)
        return d_model**-0.5 * min(step**-0.5, step * warmup_steps**-1.5)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
