from __future__ import annotations

import torch
import torch.nn.functional as F

from laboratory.losses.classification import classification_cross_entropy


def test_classification_cross_entropy_matches_torch():
    logits = torch.randn(5, 3)
    targets = torch.tensor([0, 2, 1, 0, 2])
    assert torch.allclose(
        classification_cross_entropy(logits, targets), F.cross_entropy(logits, targets)
    )


def test_classification_cross_entropy_rejects_rank_three_logits():
    try:
        classification_cross_entropy(torch.randn(2, 3, 4), torch.zeros(2, 3))
    except ValueError as error:
        assert "batch" in str(error)
    else:
        raise AssertionError("expected ValueError")
