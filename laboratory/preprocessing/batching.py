"""Batching: turn examples into the tensors a training step consumes.

* :func:`pad_sequences` — right padding with lengths.
* :class:`Seq2SeqCollator` — builds ``<bos> ... <eos>`` decoder inputs and
  ``... <eos> <pad>`` labels for teacher forcing.
* :func:`random_lm_batch` — random windows from a flat token stream.

Padding is always on the right, so no pad ever sits between real tokens. That is
what lets a causal mask ignore padding without special cases.
"""

from __future__ import annotations

from typing import Sequence

import torch

from ..utilities import lengths_mask


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


def random_lm_batch(
    ids: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random ``(x, y)`` batch from a flat token stream.

    Useful for quick experiments where building a dataset object is noise.
    """
    if len(ids) <= block_size:
        raise ValueError("token stream is shorter than block_size + 1")
    starts = torch.randint(0, len(ids) - block_size - 1, (batch_size,))
    x = torch.stack([ids[s : s + block_size] for s in starts])
    y = torch.stack([ids[s + 1 : s + block_size + 1] for s in starts])
    return x.to(device), y.to(device)


class Seq2SeqCollator:
    """Turn ``(src_ids, tgt_ids)`` pairs into the tensors a seq2seq model wants.

    Returned dict::

        src        (B, Ts)      padded source ids
        src_mask   (B, Ts)      True at real source tokens  -> encoder padding mask
        src_lengths(B,)         real source lengths
        tgt_in     (B, Tt+1)    [<bos>] + target ids        -> decoder input
        tgt_out    (B, Tt+1)    target ids + [<eos>]        -> labels
        tgt_lengths(B,)         real lengths including <eos>

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
            "src_mask": lengths_mask(src_lengths, src.size(1)),
            "src_lengths": src_lengths,
            "tgt_in": tgt_in,
            "tgt_out": tgt_out,
            "tgt_lengths": tgt_lengths,
        }


def classification_collate(
    batch: Sequence[tuple[list[int], int]],
    pad_id: int,
) -> dict[str, torch.Tensor]:
    """Pad tokenized examples and stack labels for a classifier step."""
    sequences, labels = zip(*batch)
    padded, lengths = pad_sequences(sequences, pad_id)
    return {
        "inputs": padded,
        "lengths": lengths,
        "labels": torch.tensor(labels, dtype=torch.long),
    }
