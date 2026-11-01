from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from laboratory.losses.language import token_cross_entropy


def test_token_cross_entropy_flattens_batches_and_ignores_padding():
    logits = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 2.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )
    targets = torch.tensor([[0, 1], [1, -100]])

    loss = token_cross_entropy(logits, targets)
    expected = (
        F.cross_entropy(logits[0, 0], torch.tensor(0))
        + F.cross_entropy(logits[0, 1], torch.tensor(1))
        + F.cross_entropy(logits[1, 0], torch.tensor(1))
    ) / 3  # the ignored position is excluded from the mean, not counted as zero

    assert torch.allclose(loss, expected, atol=1e-6)


def test_token_cross_entropy_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="implies targets"):
        token_cross_entropy(torch.randn(2, 3, 4), torch.zeros(2, 2, dtype=torch.long))
