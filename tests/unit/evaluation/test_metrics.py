from __future__ import annotations

import math

import pytest
import torch

from laboratory.evaluation.metrics import accuracy, exact_match, perplexity


def test_accuracy_ignores_padding_positions():
    logits = torch.tensor([[[0.0, 1.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]]])
    targets = torch.tensor([[1, 0, 0, -100]])
    assert accuracy(logits, targets, ignore_index=-100) == 1.0


def test_accuracy_of_a_fully_ignored_batch_is_zero():
    logits = torch.randn(1, 2, 3)
    targets = torch.full((1, 2), -100)
    assert accuracy(logits, targets) == 0.0


def test_perplexity_is_exponential_and_clamped():
    assert perplexity(0.0) == 1.0
    assert perplexity(math.log(5.0)) == pytest.approx(5.0)
    assert perplexity(30.0, ceiling=20.0) == pytest.approx(math.exp(20.0))


def test_exact_match_counts_only_identical_sequences():
    assert exact_match([[1, 2], [3]], [[1, 2], [3, 4]]) == 0.5
    assert exact_match([[1]], [[1]]) == 1.0
    assert exact_match([], []) == 0.0


def test_exact_match_rejects_mismatched_counts():
    with pytest.raises(ValueError, match="predictions"):
        exact_match([[1]], [])
