"""Optimizers.

Experiments choose an optimizer by name so the choice is visible in the
experiment's config instead of being buried in a constructor call.
"""

from __future__ import annotations

from typing import Iterable

import torch


def make_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    name: str = "adamw",
    learning_rate: float = 3e-4,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """Build ``"adamw"``, ``"adam"`` or ``"sgd"`` with the given settings."""
    parameters = list(parameters)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay
        )
    raise ValueError(f"unknown optimizer {name!r}; choose from 'adamw', 'adam', 'sgd'")
