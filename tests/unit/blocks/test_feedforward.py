from __future__ import annotations

import torch

from laboratory.blocks.feedforward import FeedForward, SwiGLU


def test_feed_forward_and_swiglu_preserve_shape():
    x = torch.randn(2, 5, 32)
    assert FeedForward(32, 64)(x).shape == x.shape
    assert SwiGLU(32, 48)(x).shape == x.shape
    assert SwiGLU(32)(x).shape == x.shape
