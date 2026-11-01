from __future__ import annotations

import math

import torch

from laboratory.blocks.positional_encoding import (
    ALiBi,
    LearnedPositionalEmbedding,
    RotaryEmbedding,
    SinusoidalPositionalEncoding,
    alibi_slopes,
    apply_rotary,
)


def test_sinusoidal_encoding_matches_the_formula():
    d_model, max_len = 8, 6
    encoding = SinusoidalPositionalEncoding(d_model, max_len)
    x = torch.zeros(1, max_len, d_model)
    output = encoding(x)[0]  # (max_len, d_model) after dropping the batch dim
    frequencies = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    for position in range(max_len):
        expected = torch.zeros(d_model)
        expected[0::2] = torch.sin(torch.tensor(position, dtype=torch.float32) * frequencies)
        expected[1::2] = torch.cos(torch.tensor(position, dtype=torch.float32) * frequencies)
        assert torch.allclose(output[position], expected, atol=1e-6)


def test_learned_positional_embedding_adds_the_right_vector():
    encoding = LearnedPositionalEmbedding(max_len=5, d_model=4)
    x = torch.zeros(2, 3, 4)
    output = encoding(x)
    assert torch.allclose(output[0, 2], encoding.embedding.weight[2])


def test_rotary_embedding_preserves_vector_norm():
    rope = RotaryEmbedding(head_dim=8, max_len=16)
    x = torch.randn(2, 3, 10, 8)
    cosine, sine = rope(seq_len=10)
    rotated = apply_rotary(x, cosine, sine)
    assert torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_rotary_embedding_dot_products_depend_only_on_relative_position():
    rope = RotaryEmbedding(head_dim=8, max_len=32)
    # The same base vectors at every position: the property is about the
    # *positions*, not about per-position random content.
    query = torch.randn(1, 1, 1, 8).expand(1, 1, 12, 8)
    key = torch.randn(1, 1, 1, 8).expand(1, 1, 12, 8)
    cosine, sine = rope(seq_len=12)
    rotated_query = apply_rotary(query, cosine, sine)
    rotated_key = apply_rotary(key, cosine, sine)

    scores = rotated_query[0, 0] @ rotated_key[0, 0].T  # (12, 12)
    for offset in range(12):
        diagonal = torch.diagonal(scores, offset=offset)
        assert torch.allclose(diagonal, diagonal.new_full(diagonal.shape, diagonal[0]), atol=1e-5)


def test_alibi_slopes_have_one_value_per_head_for_any_head_count():
    assert alibi_slopes(8).shape == (8,)
    assert alibi_slopes(3).shape == (3,)
    assert torch.all(alibi_slopes(4) > 0)


def test_alibi_bias_penalises_distance():
    alibi = ALiBi(n_heads=2, max_len=16)
    bias = alibi(seq_len_q=4, seq_len_k=4)[0, 0]  # (4, 4)
    assert bias[0, 0].item() == 0.0
    assert bias[3, 0] < bias[1, 0] < 0
    assert bias[0, 3] <= 0  # future positions are never rewarded
