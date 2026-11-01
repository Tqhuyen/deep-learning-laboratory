from __future__ import annotations

import torch

from laboratory.datasets.language import LanguageModelDataset, train_val_split


def test_language_model_dataset_shifts_targets_by_one():
    ids = torch.arange(10)
    dataset = LanguageModelDataset(ids, block_size=4)
    inputs, targets = dataset[0]
    assert inputs.tolist() == [0, 1, 2, 3]
    assert targets.tolist() == [1, 2, 3, 4]


def test_language_model_dataset_rejects_short_streams():
    try:
        LanguageModelDataset(torch.arange(4), block_size=4)
    except ValueError as error:
        assert "block_size" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_train_val_split_is_contiguous():
    ids = torch.arange(100)
    training, validation = train_val_split(ids, val_fraction=0.2)
    assert len(training) == 80
    assert torch.equal(training[-1] + 1, validation[0])
