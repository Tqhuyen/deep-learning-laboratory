"""Datasets and batching.

Two shapes of problem are covered, which is where almost every paper starts:

* **Language modelling** — a flat stream of ids, cut into ``(x, y)`` pairs where
  ``y`` is ``x`` shifted by one. Covers GPT-style "predict the next token".
* **Sequence-to-sequence** — aligned source/target id lists. Covers translation,
  summarisation, speech recognition. The collator builds the
  ``<bos> ... <eos>`` input and the ``... <eos> <pad>`` label tensor used with
  teacher forcing.

Both use right padding, so padding never sits between real tokens. That matters:
with a causal mask, real tokens then never attend pad positions.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset

from .tokenization import CharTokenizer, WordTokenizer

TokenPair = tuple[str, str]


def pad_sequences(
    sequences: Sequence[Sequence[int]],
    pad_id: int = 0,
    max_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of id sequences.

    Returns ``(padded, lengths)`` where ``padded`` is ``(B, T)`` and ``lengths``
    holds the pre-padding lengths. Longer sequences are truncated to ``max_len``.
    """
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    if max_len is None:
        max_len = int(lengths.max()) if len(lengths) else 0
    padded = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for row, seq in enumerate(sequences):
        n = min(len(seq), max_len)
        if n:
            padded[row, :n] = torch.as_tensor(seq[:n], dtype=torch.long)
    return padded, lengths.clamp(max=max_len)


def train_val_split(ids: torch.Tensor, val_fraction: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Contiguous split — the validation text is the tail of the corpus."""
    n_val = int(len(ids) * val_fraction)
    return ids[:-n_val], ids[-n_val:]


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


def random_lm_batch(
    ids: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random ``(x, y)`` batch from a flat token stream.

    Useful for quick experiments where building a :class:`Dataset` is noise.
    """
    if len(ids) <= block_size:
        raise ValueError("token stream is shorter than block_size + 1")
    starts = torch.randint(0, len(ids) - block_size - 1, (batch_size,))
    x = torch.stack([ids[s : s + block_size] for s in starts])
    y = torch.stack([ids[s + 1 : s + block_size + 1] for s in starts])
    return x.to(device), y.to(device)


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


class Seq2SeqCollator:
    """Turn ``(src_ids, tgt_ids)`` pairs into the tensors a seq2seq model wants.

    Returned dict::

        src        (B, Ts)      padded source ids
        src_mask   (B, Ts)      True at real source tokens  -> encoder padding mask
        src_lengths(B,)         real source lengths
        tgt_in     (B, Tt+1)    [<bos>] + target ids        -> decoder input
        tgt_out    (B, Tt+1)    target ids + [<eos>]        -> labels
        tgt_lengths(B,)         Tt + 1

    Teacher forcing: at step ``t`` the decoder is fed the true token ``t-1`` and
    asked to predict ``t``. ``tgt_in``/``tgt_out`` are the same sequence shifted
    by one, prepared here so the model never has to think about it.
    """

    def __init__(
        self,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        max_src_len: int | None = None,
        max_tgt_len: int | None = None,
    ):
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __call__(self, batch: Sequence[tuple[list[int], list[int]]]) -> dict[str, torch.Tensor]:
        src_seqs, tgt_seqs = zip(*batch)
        src, src_lengths = pad_sequences(src_seqs, self.pad_id, self.max_src_len)

        # Build <bos>-prefixed inputs and <eos>-suffixed labels *before* padding,
        # so no pad ever sits between real tokens and <eos>.
        decoder_inputs = [(self.bos_id, *ids) for ids in tgt_seqs]
        labels = [(*ids, self.eos_id) for ids in tgt_seqs]
        limit = self.max_tgt_len + 1 if self.max_tgt_len is not None else None
        tgt_in, tgt_lengths = pad_sequences(decoder_inputs, self.pad_id, limit)
        tgt_out, _ = pad_sequences(labels, self.pad_id, limit)

        return {
            "src": src,
            "src_mask": src_lengths[:, None] > torch.arange(src.size(1))[None, :],
            "src_lengths": src_lengths,
            "tgt_in": tgt_in,
            "tgt_out": tgt_out,
            "tgt_lengths": tgt_lengths,
        }
