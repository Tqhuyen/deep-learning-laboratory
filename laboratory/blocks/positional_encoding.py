"""Positional information.

Self-attention is permutation equivariant: it has no idea about order. Every
transformer therefore injects position somehow. Four schemes, each one a
distinct idea:

* :class:`SinusoidalPositionalEncoding` — fixed sin/cos waves added to the
  embeddings (Vaswani et al., 2017). Extrapolates in principle, no parameters.
* :class:`LearnedPositionalEmbedding` — a lookup table, one vector per position
  (BERT/GPT-2). Cannot extrapolate past ``max_len``.
* :class:`RotaryEmbedding` — rotates q/k pairs by a position-dependent angle;
  attention then depends only on the *relative* offset (Su et al., 2021 → RoPE,
  used by LLaMA and most modern LLMs).
* :class:`ALiBi` — adds a distance penalty straight into the attention scores
  (Press et al., 2022). No parameters, no extra embeddings.

Shapes: embeddings are ``(B, T, D)``; RoPE/ALiBi act on ``(B, H, T, D_head)``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    r"""Fixed sin/cos position vectors, added to token embeddings.

    $$PE_{(pos, 2i)}   = \sin\left(pos / 10000^{2i/d}\right)$$
    $$PE_{(pos, 2i+1)} = \cos\left(pos / 10000^{2i/d}\right)$$

    Different frequencies give the model a continuous "clock" at every scale:
    high-frequency dims distinguish neighbouring positions, low-frequency dims
    separate distant ones.
    """

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        positions = torch.arange(max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"sequence length {x.size(1)} exceeds max_len {self.pe.size(1)}")
        return self.dropout(x + self.pe[:, : x.size(1)].to(x.dtype))


class LearnedPositionalEmbedding(nn.Module):
    """One trainable vector per position, added to the embeddings."""

    def __init__(self, max_len: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_len {self.max_len}")
        positions = torch.arange(seq_len, device=x.device)
        return self.dropout(x + self.embedding(positions).to(x.dtype))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """``[x1, x2] -> [-x2, x1]`` on the last dimension (the RoPE rotation)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` (..., T, D) by the angles ``(cos, sin)`` of shape (T, D)."""
    cos = cos.to(device=x.device, dtype=x.dtype)
    sin = sin.to(device=x.device, dtype=x.dtype)
    return x * cos + rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    r"""Rotary position embedding (RoPE).

    Positions are encoded by rotating each 2-d pair of the head vector by an
    angle proportional to the position. Because rotation is linear and
    angles add, the dot product of two rotated vectors only depends on
    ``m - n`` — relative position comes for free, and nothing is added to the
    residual stream.

    Usage: build one per model with ``head_dim = d_model // n_heads``, then call
    :meth:`forward` for ``(cos, sin)`` and :func:`apply_rotary` on q and k.
    """

    def __init__(self, head_dim: int, max_len: int = 4096, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE needs an even head_dim, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim
        self.max_len = max_len

    def forward(
        self,
        seq_len: int,
        device: torch.device | str | None = None,
        offset: int = 0,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` of shape ``(seq_len, head_dim)``."""
        if offset + seq_len > self.max_len:
            raise ValueError(f"positions {offset}..{offset + seq_len} exceed max_len {self.max_len}")
        positions = torch.arange(offset, offset + seq_len, device=device or self.inv_freq.device)
        freqs = torch.outer(positions.float(), self.inv_freq)  # (T, D/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, D)
        if dtype is not None:
            emb = emb.to(dtype)
        return emb.cos(), emb.sin()


def alibi_slopes(n_heads: int) -> torch.Tensor:
    """The geometric slope sequence from the ALiBi paper, shape ``(n_heads,)``."""

    def _power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start**i for i in range(n)]

    if math.log2(n_heads).is_integer():
        slopes = _power_of_2(n_heads)
    else:
        closest = 2 ** math.floor(math.log2(n_heads))
        slopes = _power_of_2(closest) + _power_of_2(2 * closest)[0::2][: n_heads - closest]
    return torch.tensor(slopes, dtype=torch.float32)


class ALiBi(nn.Module):
    r"""Attention with linear biases: a distance penalty added to scores.

    $$ \text{bias}_{h,i,j} = -m_h \cdot (i - j) $$

    where ``m_h`` is a fixed, head-specific slope. Nearby keys are penalised
    little, distant keys a lot, so heads get different effective context widths
    without any learned parameters.
    """

    def __init__(self, n_heads: int, max_len: int = 4096):
        super().__init__()
        self.register_buffer("slopes", alibi_slopes(n_heads), persistent=False)
        self.max_len = max_len

    def forward(
        self,
        seq_len_q: int,
        seq_len_k: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return additive bias ``(1, H, Tq, Tk)`` for attention scores."""
        seq_len_k = seq_len_k or seq_len_q
        if max(seq_len_q, seq_len_k) > self.max_len:
            raise ValueError(f"sequence length exceeds max_len {self.max_len}")
        distance = torch.arange(seq_len_q, device=device)[:, None] - torch.arange(
            seq_len_k, device=device
        )[None, :]
        # Positions in the future are masked anyway; clamping keeps the bias finite.
        distance = distance.clamp(min=0).float()
        bias = -self.slopes[:, None, None].to(distance.device) * distance  # (H, Tq, Tk)
        return bias.unsqueeze(0).to(dtype=dtype)
