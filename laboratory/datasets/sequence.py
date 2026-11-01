"""Sequence-to-sequence datasets.

Aligned source/target strings are tokenized once, up front. Batching and
teacher-forcing tensors are the job of
:class:`laboratory.preprocessing.batching.Seq2SeqCollator`.
"""

from __future__ import annotations

from typing import Iterable

from torch.utils.data import Dataset

from ..preprocessing.text import CharTokenizer, WordTokenizer

TokenPair = tuple[str, str]


class Seq2SeqDataset(Dataset):
    """Pairs of raw strings tokenized once, up front."""

    def __init__(
        self,
        pairs: Iterable[TokenPair],
        src_tokenizer: CharTokenizer | WordTokenizer,
        tgt_tokenizer: CharTokenizer | WordTokenizer | None = None,
        max_src_len: int | None = None,
        max_tgt_len: int | None = None,
    ):
        tgt_tokenizer = tgt_tokenizer or src_tokenizer
        self.examples: list[tuple[list[int], list[int]]] = []
        for src, tgt in pairs:
            src_ids = src_tokenizer.encode(src)
            tgt_ids = tgt_tokenizer.encode(tgt)
            if max_src_len is not None:
                src_ids = src_ids[:max_src_len]
            if max_tgt_len is not None:
                tgt_ids = tgt_ids[:max_tgt_len]
            self.examples.append((src_ids, tgt_ids))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        return self.examples[index]
