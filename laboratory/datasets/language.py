"""Language-model datasets.

A corpus is one flat stream of token ids. Training examples are windows of that
stream with the next-token target shifted by one; the split is contiguous so the
validation text is genuinely unseen.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset


class LanguageModelDataset(Dataset):
    """Windows of ``block_size`` tokens with next-token targets.

    ``x = ids[s : s+B]`` and ``y = ids[s+1 : s+B+1]``. ``stride`` controls
    overlap between windows; the default (``block_size``) produces
    non-overlapping windows.
    """

    def __init__(self, ids: Sequence[int] | torch.Tensor, block_size: int, stride: int | None = None):
        self.ids = torch.as_tensor(ids, dtype=torch.long)
        if len(self.ids) <= block_size:
            raise ValueError(f"need at least block_size+1={block_size + 1} tokens, got {len(self.ids)}")
        self.block_size = block_size
        self.stride = stride or block_size
        self.starts = list(range(0, len(self.ids) - block_size, self.stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[index]
        x = self.ids[start : start + self.block_size]
        y = self.ids[start + 1 : start + self.block_size + 1]
        return x, y


def train_val_split(ids: torch.Tensor, val_fraction: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Contiguous split — the validation text is the tail of the corpus."""
    n_val = int(len(ids) * val_fraction)
    return ids[:-n_val], ids[-n_val:]
