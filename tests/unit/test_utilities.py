from __future__ import annotations

import pytest
import torch

from laboratory.utilities import count_parameters, get_device, lengths_mask


def test_auto_device_prefers_cuda_when_present():
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert get_device("auto").type == expected


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available on this machine")
def test_device_falls_back_to_cpu_without_cuda():
    assert get_device("cuda").type == "cpu"


def test_count_parameters_can_ignore_frozen_weights():
    model = torch.nn.Linear(3, 2)
    model.bias.requires_grad_(False)
    assert count_parameters(model) == 6
    assert count_parameters(model, trainable_only=False) == 8


def test_lengths_mask_marks_real_tokens():
    mask = lengths_mask(torch.tensor([2, 0]), max_len=3)
    assert mask.tolist() == [[True, True, False], [False, False, False]]
