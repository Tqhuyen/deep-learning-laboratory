"""Feed-forward blocks.

A transformer block is attention plus one of these. Papers mostly swap the
activation or add a gate, so both variants share the same interface
``(batch, sequence, d_model) -> (batch, sequence, d_model)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """Position-wise MLP: ``Linear(d, d_ff) -> activation -> Linear(d_ff, d)``."""

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


class SwiGLU(nn.Module):
    """Gated feed-forward used by PaLM/LLaMA.

    ``down(silu(gate(x)) * up(x))``. The gate lets the network scale features
    multiplicatively instead of only nonlinearly. Hidden size defaults to
    ``8/3 * d_model`` (same parameter count as a 4x GELU MLP).
    """

    def __init__(self, d_model: int, hidden: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden = hidden or int(8 * d_model / 3)
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


def get_activation(name: str):
    activations = {"gelu": F.gelu, "relu": F.relu, "silu": F.silu, "tanh": torch.tanh}
    if name not in activations:
        raise ValueError(f"unknown activation {name!r}; choose from {sorted(activations)}")
    return activations[name]


def make_ffn(name: str, d_model: int, d_ff: int | None, dropout: float):
    """Factory for ``"mlp"`` (GELU MLP) or ``"swiglu"``."""
    if name == "mlp":
        return FeedForward(d_model, d_ff, dropout)
    if name == "swiglu":
        return SwiGLU(d_model, d_ff, dropout)
    raise ValueError(f"unknown ffn {name!r}")
