from __future__ import annotations

import math

import pytest
import torch

from laboratory.attention import (
    MultiHeadAttention,
    repeat_kv,
    scaled_dot_product_attention,
)
from laboratory.utilities import causal_mask


def reference_attention(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


def test_scaled_dot_product_attention_matches_manual_computation():
    q, k, v = torch.randn(2, 3, 5, 8), torch.randn(2, 3, 6, 8), torch.randn(2, 3, 6, 8)
    mask = torch.ones(5, 6, dtype=torch.bool).tril()
    output = scaled_dot_product_attention(q, k, v, attn_mask=mask)
    expected, _ = reference_attention(q, k, v, mask)
    assert torch.allclose(output, expected, atol=1e-6)


def test_attention_weights_are_zero_where_masked():
    q, k, v = torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8)
    _, weights = scaled_dot_product_attention(
        q, k, v, attn_mask=causal_mask(4), return_weights=True
    )
    assert torch.allclose(weights[0, 0].triu(diagonal=1), torch.zeros(4, 4))


def test_causal_attention_does_not_leak_the_future():
    attention = MultiHeadAttention(d_model=16, n_heads=2, dropout=0.0)
    x = torch.randn(1, 5, 16)
    modified = x.clone()
    modified[:, -1] += 10.0
    with torch.no_grad():
        original_output = attention(x, is_causal=True)
        modified_output = attention(modified, is_causal=True)
    assert torch.allclose(original_output[:, :-1], modified_output[:, :-1], atol=1e-6)
    assert not torch.allclose(original_output[:, -1], modified_output[:, -1], atol=1e-6)


def test_padding_mask_hides_padded_keys():
    attention = MultiHeadAttention(d_model=8, n_heads=2, dropout=0.0)
    x = torch.randn(1, 4, 8)
    modified = x.clone()
    modified[:, 3] = 99.0  # this position is declared padding
    padding = torch.tensor([[True, True, True, False]])
    with torch.no_grad():
        original_output, original_weights = attention(
            x, key_padding_mask=padding, return_weights=True
        )
        modified_output, modified_weights = attention(
            modified, key_padding_mask=padding, return_weights=True
        )
    assert torch.allclose(original_output[:, :3], modified_output[:, :3], atol=1e-6)
    assert original_weights[..., 3].abs().max() < 1e-6


def test_grouped_query_attention_repeats_key_value_heads():
    x = torch.randn(2, 3, 4, 6)
    repeated = repeat_kv(x, n_rep=2)
    assert repeated.shape == (2, 6, 4, 6)
    assert torch.allclose(repeated[:, 0], repeated[:, 1])


def test_grouped_query_attention_forward_shape():
    attention = MultiHeadAttention(d_model=32, n_heads=4, n_kv_heads=2, dropout=0.0)
    output = attention(torch.randn(2, 7, 32), is_causal=True)
    assert output.shape == (2, 7, 32)


def test_cross_attention_uses_the_memory_length():
    attention = MultiHeadAttention(d_model=16, n_heads=2, dropout=0.0)
    query = torch.randn(2, 3, 16)
    memory = torch.randn(2, 9, 16)
    output, weights = attention(query, memory=memory, return_weights=True)
    assert output.shape == (2, 3, 16)
    assert weights.shape == (2, 2, 3, 9)


def test_causal_attention_rejects_different_query_and_key_lengths():
    attention = MultiHeadAttention(d_model=8, n_heads=1)
    with pytest.raises(ValueError, match="causal attention needs"):
        attention(torch.randn(1, 3, 8), memory=torch.randn(1, 5, 8), is_causal=True)
