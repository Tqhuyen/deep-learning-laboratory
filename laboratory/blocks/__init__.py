"""Reusable building blocks.

Each module here is a component you can drop into a model: attention,
positional encodings, normalisation and feed-forward blocks. Assembled
architectures live in :mod:`laboratory.models`; losses live in
:mod:`laboratory.objectives`.

The rule for adding code: a component that is useful to more than one model
family goes here, with the same forward signature as what it replaces.
"""

from .attention import MultiHeadAttention, repeat_kv, scaled_dot_product_attention
from .feedforward import FeedForward, SwiGLU, get_activation, make_ffn
from .normalization import LayerNorm, RMSNorm, make_norm
from .positional_encoding import (
    ALiBi,
    LearnedPositionalEmbedding,
    RotaryEmbedding,
    SinusoidalPositionalEncoding,
    apply_rotary,
)

__all__ = [
    "ALiBi",
    "FeedForward",
    "LayerNorm",
    "LearnedPositionalEmbedding",
    "MultiHeadAttention",
    "RMSNorm",
    "RotaryEmbedding",
    "SinusoidalPositionalEncoding",
    "SwiGLU",
    "apply_rotary",
    "get_activation",
    "make_ffn",
    "make_norm",
    "repeat_kv",
    "scaled_dot_product_attention",
]
