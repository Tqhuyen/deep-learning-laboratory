from __future__ import annotations

import torch

from laboratory.training.metrics import RunningMean, accuracy


def test_accuracy_ignores_padding_positions():
    logits = torch.tensor([[[0.0, 1.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]]])
    targets = torch.tensor([[1, 0, 0, -100]])
    assert accuracy(logits, targets, ignore_index=-100) == 1.0


def test_accuracy_of_a_fully_ignored_batch_is_zero():
    logits = torch.randn(1, 2, 3)
    targets = torch.full((1, 2), -100)
    assert accuracy(logits, targets) == 0.0


def test_running_mean_is_weighted_by_batch_size():
    mean = RunningMean()
    mean.update(2.0, n=3)
    mean.update(4.0, n=1)
    assert mean.mean == 2.5
