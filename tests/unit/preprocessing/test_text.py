from __future__ import annotations

import pytest

from laboratory.preprocessing.text import (
    BPETokenizer,
    CharTokenizer,
    WordTokenizer,
    load_tokenizer,
    save_tokenizer,
)


def test_character_tokenizer_round_trips_unicode_and_newlines():
    text = "héllo\nworld ✓"
    tokenizer = CharTokenizer.from_text(text)
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_character_tokenizer_uses_unknown_token_when_present():
    tokenizer = CharTokenizer.from_text("abc", special_tokens=["<unk>"])
    assert tokenizer.encode("a z")[1] == tokenizer.unk_id


def test_character_tokenizer_rejects_unknown_characters_without_unk():
    tokenizer = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError, match="unknown character"):
        tokenizer.encode("abz")


def test_word_tokenizer_lowercases_and_maps_rare_words_to_unk():
    tokenizer = WordTokenizer.from_text(
        "The cat sat. The cat ran.", special_tokens=["<unk>"], min_freq=1
    )
    ids = tokenizer.encode("the cat flew")
    assert ids[:2] == tokenizer.encode("the cat")[:2]
    assert ids[2] == tokenizer.unk_id
    assert tokenizer.decode(tokenizer.encode("The cat sat.")) == "the cat sat ."


def test_bpe_training_produces_merges_and_round_trips_words():
    text = "low low low low lower lower newest newest widest widest"
    tokenizer = BPETokenizer.train(text, vocab_size=40)
    assert tokenizer.merges, "training should learn at least one merge"
    assert tokenizer.vocab_size <= 40
    sentence = "low lower newest widest"
    assert tokenizer.decode(tokenizer.encode(sentence)) == sentence
    # A merge reduces the number of tokens relative to raw characters.
    assert len(tokenizer.encode("lowest")) <= len("lowest") + 1


def test_bpe_encoding_is_deterministic():
    tokenizer = BPETokenizer.train("abab abab abab cdcd", vocab_size=20)
    assert tokenizer.encode("abab") == tokenizer.encode("abab")


def test_tokenizers_survive_a_disk_round_trip(tmp_path):
    character = CharTokenizer.from_text("abc", special_tokens=["<unk>"])
    save_tokenizer(character, tmp_path / "character.json")
    restored = load_tokenizer(tmp_path / "character.json")
    assert restored.decode(restored.encode("cab")) == "cab"
    assert restored.vocab_size == character.vocab_size

    bpe = BPETokenizer.train("the quick brown fox jumps over the lazy dog", vocab_size=40)
    save_tokenizer(bpe, tmp_path / "bpe.json")
    restored_bpe = load_tokenizer(tmp_path / "bpe.json")
    assert restored_bpe.encode("the quick fox") == bpe.encode("the quick fox")
