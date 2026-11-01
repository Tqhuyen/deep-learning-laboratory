from __future__ import annotations

import torch
import torch.nn.functional as F

from laboratory.layers import FeedForward, LayerNorm, RMSNorm, SwiGLU


def test_layer_norm_matches_torch_reference():
    norm = LayerNorm(16)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(16))
        norm.bias.copy_(torch.randn(16))
    x = torch.randn(4, 5, 16)
    expected = F.layer_norm(x, (16,), norm.weight, norm.bias, eps=1e-5)
    assert torch.allclose(norm(x), expected, atol=1e-6)


def test_rms_norm_matches_its_formula():
    norm = RMSNorm(12)
    with torch.no_grad():
        norm.weight.copy_(torch.rand(12) + 0.5)
    x = torch.randn(3, 7, 12)
    expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * norm.weight
    assert torch.allclose(norm(x), expected, atol=1e-6)


def test_feed_forward_and_swiglu_preserve_shape():
    x = torch.randn(2, 5, 32)
    assert FeedForward(32, 64)(x).shape == x.shape
    assert SwiGLU(32, 48)(x).shape == x.shape
    assert SwiGLU(32)(x).shape == x.shape
