from __future__ import annotations

from laboratory.datasets.sentiment import (
    SentimentDataset,
    read_sentiment_tsv,
    stratified_split,
)
from laboratory.preprocessing.text import WordTokenizer


def test_read_sentiment_tsv_accepts_header_and_raw_rows(tmp_path):
    path = tmp_path / "sentiment.tsv"
    path.write_text("sentence\tlabel\ngood film\t1\nbad film\t0\n", encoding="utf-8")
    assert read_sentiment_tsv(path) == [("good film", 1), ("bad film", 0)]


def test_stratified_split_keeps_classes_on_both_sides():
    examples = [(f"good {index}", 1) for index in range(20)] + [
        (f"bad {index}", 0) for index in range(20)
    ]
    train, test = stratified_split(examples, test_fraction=0.25, seed=0)
    assert len(train) == 30 and len(test) == 10
    assert sum(label for _, label in train) == 15
    assert sum(label for _, label in test) == 5
    assert not set(train) & set(test)


def test_sentiment_dataset_truncates_and_keeps_labels():
    tokenizer = WordTokenizer.from_text(
        "a b c d", special_tokens=["<pad>", "<unk>"], min_freq=1
    )
    dataset = SentimentDataset([("a b c d", 1)], tokenizer, max_len=2)
    ids, label = dataset[0]
    assert ids == tokenizer.encode("a b") and label == 1
