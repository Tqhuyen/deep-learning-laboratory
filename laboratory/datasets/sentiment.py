"""Sentiment datasets: TSV loading, stratified splits, tokenized examples."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

from torch.utils.data import Dataset

from ..preprocessing.text import CharTokenizer, WordTokenizer

Example = tuple[str, int]


def read_sentiment_tsv(path: str | Path) -> list[Example]:
    """Read ``sentence<TAB>label`` rows, skipping an optional header."""
    examples: list[Example] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.rsplit("\t", 1)
        if len(parts) != 2:
            raise ValueError(f"expected two tab-separated columns in {path}: {line!r}")
        sentence, label = parts
        if sentence.strip().lower() in ("sentence", "text"):
            continue
        examples.append((sentence.strip(), int(label)))
    return examples


def stratified_split(
    examples: Sequence[Example],
    test_fraction: float = 0.25,
    seed: int = 0,
) -> tuple[list[Example], list[Example]]:
    """Split each class separately so both sides keep the label balance."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    generator = random.Random(seed)
    by_label: dict[int, list[int]] = {}
    for index, (_, label) in enumerate(examples):
        by_label.setdefault(label, []).append(index)

    test_indices: set[int] = set()
    for indices in by_label.values():
        generator.shuffle(indices)
        take = max(1, int(len(indices) * test_fraction))
        test_indices.update(indices[:take])

    train = [example for index, example in enumerate(examples) if index not in test_indices]
    test = [example for index, example in enumerate(examples) if index in test_indices]
    return train, test


class SentimentDataset(Dataset):
    """Tokenized ``(ids, label)`` examples with truncation."""

    def __init__(
        self,
        examples: Iterable[Example],
        tokenizer: CharTokenizer | WordTokenizer,
        max_len: int = 64,
    ):
        self.items: list[tuple[list[int], int]] = []
        for text, label in examples:
            ids = tokenizer.encode(text)[:max_len]
            if ids:
                self.items.append((ids, int(label)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.items[index]
