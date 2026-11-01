"""Transformer models assembled from the pieces above.

Three models, nothing else:

* :class:`TransformerEncoder` — bidirectional encoder (BERT-style, or the
  encoder half of Vaswani et al.).
* :class:`TransformerDecoder` — causal decoder with cross-attention.
* :class:`GPT` — decoder-only causal LM (GPT/LLaMA-style).
* :class:`Seq2SeqTransformer` — encoder + decoder + output projection, i.e. the
  original 2017 architecture; decoding lives in laboratory.inference.

Pre-LN (``norm_first=True``, the modern default) normalises *before* each
sublayer and adds a final norm; post-LN matches the original paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..blocks import MultiHeadAttention, make_ffn, make_norm
from ..blocks.positional_encoding import (
    LearnedPositionalEmbedding,
    RotaryEmbedding,
    SinusoidalPositionalEncoding,
)
from ..losses.language import token_cross_entropy


def build_positional(name: str, d_model: int, max_len: int, dropout: float):
    """``"learned"``, ``"sinusoidal"``, or ``"rope"``/``"none"`` (handled in attention)."""
    if name == "learned":
        return LearnedPositionalEmbedding(max_len, d_model, dropout)
    if name == "sinusoidal":
        return SinusoidalPositionalEncoding(d_model, max_len, dropout)
    if name in ("rope", "none"):
        return None
    raise ValueError(f"unknown positional encoding {name!r}")


def build_rope(pos: str, d_model: int, n_heads: int, max_len: int) -> RotaryEmbedding | None:
    if pos != "rope":
        return None
    if (d_model // n_heads) % 2:
        raise ValueError("RoPE requires an even head_dim: d_model / n_heads must be even")
    return RotaryEmbedding(d_model // n_heads, max_len=max_len)


class EncoderLayer(nn.Module):
    """Self-attention + feed-forward, residual around both.

    With ``causal=True`` this layer *is* a GPT block, which is why the decoder
    in :class:`GPT` reuses it.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int | None = None,
        dropout: float = 0.0,
        norm_first: bool = True,
        norm: str = "layernorm",
        ffn: str = "mlp",
        n_kv_heads: int | None = None,
        bias: bool = True,
        causal: bool = False,
        rope: RotaryEmbedding | None = None,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model, n_heads, n_kv_heads=n_kv_heads, dropout=dropout, bias=bias, rope=rope
        )
        self.ffn = make_ffn(ffn, d_model, d_ff, dropout)
        self.norm1 = make_norm(norm, d_model)
        self.norm2 = make_norm(norm, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first
        self.causal = causal

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.norm_first:
            x = x + self.dropout(self.self_attn(
                self.norm1(x), key_padding_mask=key_padding_mask, is_causal=self.causal
            ))
            x = x + self.dropout(self.ffn(self.norm2(x)))
        else:
            x = self.norm1(x + self.dropout(self.self_attn(
                x, key_padding_mask=key_padding_mask, is_causal=self.causal
            )))
            x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    """Masked self-attention, then cross-attention over the encoder output."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int | None = None,
        dropout: float = 0.0,
        norm_first: bool = True,
        norm: str = "layernorm",
        ffn: str = "mlp",
        n_kv_heads: int | None = None,
        bias: bool = True,
        rope: RotaryEmbedding | None = None,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model, n_heads, n_kv_heads=n_kv_heads, dropout=dropout, bias=bias, rope=rope
        )
        self.cross_attn = MultiHeadAttention(
            d_model, n_heads, n_kv_heads=n_kv_heads, dropout=dropout, bias=bias
        )
        self.ffn = make_ffn(ffn, d_model, d_ff, dropout)
        self.norm1 = make_norm(norm, d_model)
        self.norm2 = make_norm(norm, d_model)
        self.norm3 = make_norm(norm, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.norm_first:
            x = x + self.dropout(self.self_attn(
                self.norm1(x), key_padding_mask=tgt_key_padding_mask, is_causal=True
            ))
            x = x + self.dropout(self.cross_attn(
                self.norm2(x), memory=memory, key_padding_mask=memory_key_padding_mask
            ))
            x = x + self.dropout(self.ffn(self.norm3(x)))
        else:
            x = self.norm1(x + self.dropout(self.self_attn(
                x, key_padding_mask=tgt_key_padding_mask, is_causal=True
            )))
            x = self.norm2(x + self.dropout(self.cross_attn(
                x, memory=memory, key_padding_mask=memory_key_padding_mask
            )))
            x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class TransformerEncoder(nn.Module):
    """Embeddings + N encoder layers."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int | None = None,
        max_len: int = 512,
        dropout: float = 0.1,
        norm_first: bool = True,
        norm: str = "layernorm",
        ffn: str = "mlp",
        n_kv_heads: int | None = None,
        pos: str = "sinusoidal",
        scale_embeddings: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = build_positional(pos, d_model, max_len, dropout)
        self.dropout = nn.Dropout(dropout)
        rope = build_rope(pos, d_model, n_heads, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, norm_first, norm, ffn,
                         n_kv_heads, bias, causal=False, rope=rope)
            for _ in range(n_layers)
        ])
        self.final_norm = make_norm(norm, d_model) if norm_first else nn.Identity()
        self.scale = d_model**0.5 if scale_embeddings else 1.0
        self.max_len = max_len

    def forward(self, ids: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.token_embedding(ids) * self.scale
        x = self.pos_encoding(x) if self.pos_encoding is not None else self.dropout(x)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return self.final_norm(x)


class TransformerDecoder(nn.Module):
    """Embeddings + N decoder layers with cross-attention."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int | None = None,
        max_len: int = 512,
        dropout: float = 0.1,
        norm_first: bool = True,
        norm: str = "layernorm",
        ffn: str = "mlp",
        n_kv_heads: int | None = None,
        pos: str = "sinusoidal",
        scale_embeddings: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = build_positional(pos, d_model, max_len, dropout)
        self.dropout = nn.Dropout(dropout)
        rope = build_rope(pos, d_model, n_heads, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, norm_first, norm, ffn,
                         n_kv_heads, bias, rope=rope)
            for _ in range(n_layers)
        ])
        self.final_norm = make_norm(norm, d_model) if norm_first else nn.Identity()
        self.scale = d_model**0.5 if scale_embeddings else 1.0
        self.max_len = max_len

    def forward(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.token_embedding(tgt_ids) * self.scale
        x = self.pos_encoding(x) if self.pos_encoding is not None else self.dropout(x)
        for layer in self.layers:
            x = layer(
                x,
                memory,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return self.final_norm(x)


class GPT(nn.Module):
    """Decoder-only causal language model.

    ``forward`` always returns ``(logits, loss)``; ``loss`` is ``None`` when no
    targets are given. Logits have shape ``(B, T, vocab_size)``.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int | None = None,
        max_len: int = 512,
        dropout: float = 0.0,
        norm_first: bool = True,
        norm: str = "layernorm",
        ffn: str = "mlp",
        n_kv_heads: int | None = None,
        pos: str = "learned",
        tie_embeddings: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = build_positional(pos, d_model, max_len, dropout)
        self.dropout = nn.Dropout(dropout)
        rope = build_rope(pos, d_model, n_heads, max_len)
        self.blocks = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, norm_first, norm, ffn,
                         n_kv_heads, bias, causal=True, rope=rope)
            for _ in range(n_layers)
        ])
        self.final_norm = make_norm(norm, d_model) if norm_first else nn.Identity()
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        if idx.size(1) > self.max_len:
            raise ValueError(f"sequence length {idx.size(1)} exceeds max_len {self.max_len}")
        x = self.token_embedding(idx)
        x = self.pos_encoding(x) if self.pos_encoding is not None else self.dropout(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))

        loss = None
        if targets is not None:
            loss = token_cross_entropy(logits, targets)
        return logits, loss


class Seq2SeqTransformer(nn.Module):
    """Encoder-decoder transformer for sequence-to-sequence tasks.

    ``forward`` returns logits ``(B, Tt, tgt_vocab_size)``; teacher forcing is
    provided by :class:`laboratory.preprocessing.Seq2SeqCollator` (``tgt_in`` / ``tgt_out``).
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int | None = None,
        max_len: int = 512,
        dropout: float = 0.1,
        norm_first: bool = True,
        norm: str = "layernorm",
        ffn: str = "mlp",
        n_kv_heads: int | None = None,
        pos: str = "sinusoidal",
        tie_embeddings: bool = False,
        bos_id: int = 1,
        eos_id: int = 2,
        pad_id: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        if tie_embeddings and src_vocab_size != tgt_vocab_size:
            raise ValueError("weight tying needs matching source and target vocabularies")
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout,
            norm_first, norm, ffn, n_kv_heads, pos, True, bias,
        )
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout,
            norm_first, norm, ffn, n_kv_heads, pos, True, bias,
        )
        self.lm_head = nn.Linear(d_model, tgt_vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.decoder.token_embedding.weight
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.max_len = max_len

    def encode(self, src_ids: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None):
        return self.encoder(src_ids, key_padding_mask=src_key_padding_mask)

    def decode(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ):
        hidden = self.decoder(
            tgt_ids,
            memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.lm_head(hidden)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory = self.encode(src_ids, src_key_padding_mask)
        return self.decode(tgt_ids, memory, src_key_padding_mask, tgt_key_padding_mask)
