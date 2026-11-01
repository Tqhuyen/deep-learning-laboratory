r"""Attention, written once, carefully.

The core is 6 lines of math:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + \text{bias}\right) V $$

Everything else in this file is plumbing that papers reuse verbatim:

* boolean masks (True = may attend) for causality and padding,
* multi-head splitting/merging (``n_heads``),
* grouped-query attention (``n_kv_heads < n_heads``, Shazeer 2019 / Ainslie
  et al. 2023) — fewer k/v heads, cheaper decoding,
* optional RoPE applied to q/k,
* cross-attention by passing ``memory`` instead of ``x``.

A note on masking: this repo uses **True = keep**, matching
``F.scaled_dot_product_attention``. If you later swap the manual implementation
for the fused kernel, the mask semantics do not change.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .positional_encoding import RotaryEmbedding, apply_rotary
from .utilities import causal_mask


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
    bias: torch.Tensor | None = None,
    return_weights: bool = False,
):
    """Softmax attention over the last two dimensions.

    Args:
        q: ``(..., Tq, D)`` queries.
        k: ``(..., Tk, D)`` keys.
        v: ``(..., Tk, D)`` values.
        attn_mask: boolean, broadcastable to ``(..., Tq, Tk)``, True = allowed.
        bias: additive bias broadcastable to the scores, e.g. ALiBi.
        return_weights: also return the ``(..., Tq, Tk)`` attention weights.

    ``torch.finfo.min`` is used instead of ``-inf`` for masked positions so a
    fully masked row yields zeros rather than NaN.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if bias is not None:
        scores = scores + bias
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask.bool(), torch.finfo(scores.dtype).min)

    # fp16 attention is numerically fragile; do the softmax in fp32 when needed.
    weights = torch.softmax(scores.float() if scores.dtype == torch.float16 else scores, dim=-1)
    weights = weights.to(scores.dtype)
    if dropout_p:
        weights = torch.dropout(weights, dropout_p, train=training)

    out = torch.matmul(weights, v)
    return (out, weights) if return_weights else out


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand ``(B, H_kv, T, D)`` to ``(B, H_q, T, D)`` by repeating each head."""
    if n_rep == 1:
        return x
    batch, n_kv, seq_len, head_dim = x.shape
    return (
        x[:, :, None]
        .expand(batch, n_kv, n_rep, seq_len, head_dim)
        .reshape(batch, n_kv * n_rep, seq_len, head_dim)
    )


class MultiHeadAttention(nn.Module):
    """Multi-head (or grouped-query, or cross) attention.

    Args:
        d_model: input/output width.
        n_heads: number of query heads.
        n_kv_heads: number of key/value heads (defaults to ``n_heads``).
            ``n_heads // n_kv_heads`` queries share each k/v head.
        rope: optional :class:`RotaryEmbedding`; rotations are applied to q/k
            after the head split.

    Mask semantics: True means "this key is visible".
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        rope: RotaryEmbedding | None = None,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        n_kv_heads = n_kv_heads or n_heads
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads={n_heads} must be a multiple of n_kv_heads={n_kv_heads}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.dropout = dropout
        self.rope = rope

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=bias)
        self.kv_proj = nn.Linear(d_model, 2 * n_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        return_weights: bool = False,
    ):
        """Attend.

        Args:
            x: queries, ``(B, Tq, d_model)``.
            memory: keys/values ``(B, Tk, d_model)`` for cross-attention;
                ``None`` means self-attention.
            attn_mask: extra boolean mask, broadcastable to ``(B, Tq, Tk)``.
            key_padding_mask: ``(B, Tk)`` True at real keys (mask out pads).
            is_causal: apply a lower-triangular mask (decoder self-attention).
            return_weights: also return attention weights ``(B, H, Tq, Tk)``.
        """
        batch, tgt_len, _ = x.shape
        kv_input = x if memory is None else memory
        src_len = kv_input.size(1)

        q = self.q_proj(x).view(batch, tgt_len, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv_proj(kv_input).chunk(2, dim=-1)
        k = k.view(batch, src_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, src_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            cos_q, sin_q = self.rope(tgt_len, device=x.device)
            q = apply_rotary(q, cos_q, sin_q)
            cos_k, sin_k = self.rope(src_len, device=x.device)
            k = apply_rotary(k, cos_k, sin_k)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        mask = None
        if is_causal:
            if tgt_len != src_len:
                raise ValueError(f"causal attention needs Tq == Tk, got {tgt_len} and {src_len}")
            mask = causal_mask(tgt_len, device=x.device)
        if attn_mask is not None:
            mask = attn_mask.bool() if mask is None else mask & attn_mask.bool()
        if key_padding_mask is not None:
            pad = key_padding_mask.bool()[:, None, None, :]
            mask = pad if mask is None else mask & pad

        out, weights = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout,
            training=self.training,
            return_weights=True,
        )
        out = out.transpose(1, 2).reshape(batch, tgt_len, self.d_model)
        out = self.out_proj(out)
        return (out, weights) if return_weights else out
