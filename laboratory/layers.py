"""Normalisation, feed-forward networks and other small pieces.

These are the parts papers swap most often (pre-LN vs post-LN, GELU vs SwiGLU,
LayerNorm vs RMSNorm), so each one is a self-contained module with the formula
in its docstring.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    r"""Layer normalisation (Ba et al., 2016), written out by hand.

    $$\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

    Statistics are taken over the last dimension per token, not over the batch.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


class RMSNorm(nn.Module):
    r"""Root-mean-square normalisation (Zhang & Sennrich, 2019; used by LLaMA).

    $$\text{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}}$$

    No mean subtraction and no bias, which is cheaper and works as well.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


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
        self.activation = _get_activation(activation)

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


def _get_activation(name: str):
    activations = {"gelu": F.gelu, "relu": F.relu, "silu": F.silu, "tanh": torch.tanh}
    if name not in activations:
        raise ValueError(f"unknown activation {name!r}; choose from {sorted(activations)}")
    return activations[name]


def make_norm(name: str, d_model: int):
    """Factory used by the transformer blocks: ``"layernorm"`` or ``"rmsnorm"``."""
    if name == "layernorm":
        return LayerNorm(d_model)
    if name == "rmsnorm":
        return RMSNorm(d_model)
    raise ValueError(f"unknown norm {name!r}")


def make_ffn(name: str, d_model: int, d_ff: int | None, dropout: float):
    """Factory for ``"mlp"`` (GELU MLP) or ``"swiglu"``."""
    if name == "mlp":
        return FeedForward(d_model, d_ff, dropout)
    if name == "swiglu":
        return SwiGLU(d_model, d_ff, dropout)
    raise ValueError(f"unknown ffn {name!r}")
