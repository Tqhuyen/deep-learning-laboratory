from __future__ import annotations

from laboratory.datasets.sequence import Seq2SeqDataset
from laboratory.preprocessing.text import WordTokenizer


def make_tokenizer() -> WordTokenizer:
    return WordTokenizer.from_text("1 2 3", special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"])


def test_sequence_dataset_tokenizes_both_sides():
    tokenizer = make_tokenizer()
    dataset = Seq2SeqDataset([("1 2", "2 1")], tokenizer)
    source, target = dataset[0]
    assert source == tokenizer.encode("1 2")
    assert target == tokenizer.encode("2 1")


def test_sequence_dataset_truncates_to_requested_lengths():
    tokenizer = make_tokenizer()
    dataset = Seq2SeqDataset(
        [("1 2 3", "1 2 3")], tokenizer, max_src_len=2, max_tgt_len=1
    )
    source, target = dataset[0]
    assert source == tokenizer.encode("1 2")
    assert target == tokenizer.encode("1")
