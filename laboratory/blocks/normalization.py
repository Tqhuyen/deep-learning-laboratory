"""Normalisation layers.

LayerNorm and RMSNorm differ only in what they divide by, so they live
together. The ``make_norm`` factory is what the transformer blocks use, which
keeps a paper's "we replace LayerNorm with X" experiment a one-line change.
"""

from __future__ import annotations

import torch
import torch.nn as nn


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


def make_norm(name: str, d_model: int):
    """Factory used by the transformer blocks: ``"layernorm"`` or ``"rmsnorm"``."""
    if name == "layernorm":
        return LayerNorm(d_model)
    if name == "rmsnorm":
        return RMSNorm(d_model)
    raise ValueError(f"unknown norm {name!r}")
