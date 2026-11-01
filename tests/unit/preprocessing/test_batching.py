from __future__ import annotations

import torch

from laboratory.datasets.sequence import Seq2SeqDataset
from laboratory.preprocessing.batching import Seq2SeqCollator, pad_sequences, random_lm_batch
from laboratory.preprocessing.text import WordTokenizer


def test_pad_sequences_reports_pre_padding_lengths_and_truncates():
    padded, lengths = pad_sequences([[1, 2, 3], [4], []], pad_id=0, max_len=3)
    assert padded.tolist() == [[1, 2, 3], [4, 0, 0], [0, 0, 0]]
    assert lengths.tolist() == [3, 1, 0]

    truncated, lengths = pad_sequences([[1, 2, 3, 4]], pad_id=9, max_len=2)
    assert truncated.tolist() == [[1, 2]]
    assert lengths.tolist() == [2]


def test_random_lm_batch_returns_shifted_pairs():
    ids = torch.arange(100)
    inputs, targets = random_lm_batch(ids, block_size=8, batch_size=5)
    assert inputs.shape == targets.shape == (5, 8)
    assert torch.equal(targets[:, :-1], inputs[:, 1:])


def test_seq2seq_collator_builds_teacher_forcing_tensors():
    tokenizer = WordTokenizer.from_text(
        "1 2 3", special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"]
    )
    dataset = Seq2SeqDataset(
        [("1 2", "2 1"), ("3", "3")], tokenizer, tgt_tokenizer=tokenizer
    )
    collator = Seq2SeqCollator(tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id)
    batch = collator([dataset[0], dataset[1]])

    assert batch["src"].shape == (2, 2)
    assert batch["src_mask"].tolist() == [[True, True], [True, False]]
    assert batch["tgt_in"].shape == (2, 3)
    assert batch["tgt_in"][0].tolist() == [tokenizer.bos_id, *tokenizer.encode("2 1")]
    assert batch["tgt_in"][1].tolist() == [tokenizer.bos_id, *tokenizer.encode("3"), tokenizer.pad_id]
    assert batch["tgt_out"][0].tolist() == [*tokenizer.encode("2 1"), tokenizer.eos_id]
    assert batch["tgt_out"][1].tolist() == [*tokenizer.encode("3"), tokenizer.eos_id, tokenizer.pad_id]
    assert batch["tgt_lengths"].tolist() == [3, 2]
