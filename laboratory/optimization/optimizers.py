"""Optimizers.

Experiments choose an optimizer by name so the choice is visible in the
experiment's config instead of being buried in a constructor call. The default
is Muon (:mod:`laboratory.optimization.muon`), which needs the model to tell
hidden matrices from embeddings and output layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .muon import Muon, muon_parameter_groups


def make_optimizer(
    model: nn.Module,
    name: str = "muon",
    learning_rate: float = 3e-4,
    weight_decay: float = 0.0,
    momentum: float | None = None,
) -> torch.optim.Optimizer:
    """Build ``"muon"``, ``"adamw"``, ``"adam"`` or ``"sgd"`` for ``model``'s parameters.

    ``momentum`` defaults to 0.95 for Muon and 0.9 for SGD.
    """
    if name == "muon":
        return Muon(
            muon_parameter_groups(model),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.95 if momentum is None else momentum,
        )
    parameters = list(model.parameters())
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=0.9 if momentum is None else momentum,
            weight_decay=weight_decay,
        )
    raise ValueError(f"unknown optimizer {name!r}; choose from 'muon', 'adamw', 'adam', 'sgd'")
