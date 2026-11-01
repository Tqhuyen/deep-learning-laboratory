"""Tokenizers: text <-> integer ids.

Every model in this repo works on ids, so this file is where a paper's
"vocabulary" becomes something concrete. Three implementations, increasing in
sophistication:

============================  ============================  ==================
class                         granularity                   decode is exact?
============================  ============================  ==================
``CharTokenizer``             one character                 yes
``WordTokenizer``             whitespace/punctuation word   no (spacing is lost)
``BPETokenizer``              subword (learned merges)      no (spacing is lost)
============================  ============================  ==================

Special tokens (``<pad>``, ``<bos>``, ``<eos>``, ``<unk>``) are ordinary
entries at the front of the vocabulary. Give them to ``from_text``/``train``
and the ``*_id`` properties pick them up.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

END_OF_WORD = "</w>"
_WORD_RE = re.compile(r"\S+")


class CharTokenizer:
    """One token per character. Smallest possible vocabulary.

    Shakespeare has ~65 unique characters, so a char model has no OOV problem
    and no tokenizer training step: the corpus *is* the vocabulary.
    """

    def __init__(self, vocab: Iterable[str], special_tokens: Sequence[str] = ()):
        self.special_tokens = list(special_tokens)
        chars = [c for c in vocab if c not in self.special_tokens]
        self.itos = self.special_tokens + chars
        self.stoi = {token: i for i, token in enumerate(self.itos)}
        self._unk = self.stoi.get("<unk>")

    @classmethod
    def from_text(cls, text: str, special_tokens: Sequence[str] = ()) -> "CharTokenizer":
        return cls(sorted(set(text)), special_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int | None:
        return self.stoi.get("<pad>")

    @property
    def bos_id(self) -> int | None:
        return self.stoi.get("<bos>")

    @property
    def eos_id(self) -> int | None:
        return self.stoi.get("<eos>")

    @property
    def unk_id(self) -> int | None:
        return self._unk

    def encode(self, text: str) -> list[int]:
        ids = []
        for char in text:
            idx = self.stoi.get(char, self._unk)
            if idx is None:
                raise ValueError(f"unknown character {char!r}; use <unk> or extend the vocab")
            ids.append(idx)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def _state(self) -> dict:
        base = [c for c in self.itos if c not in self.special_tokens]
        return {"type": "char", "vocab": base, "special_tokens": self.special_tokens}


class WordTokenizer:
    """One token per word; punctuation becomes its own token.

    Words seen fewer than ``min_freq`` times are dropped and later mapped to
    ``<unk>``. Decoding joins tokens with spaces, so the original spacing and
    newlines are not recovered.
    """

    def __init__(self, vocab: Iterable[str], special_tokens: Sequence[str] = (), lowercase: bool = True):
        self.special_tokens = list(special_tokens)
        words = [w for w in vocab if w not in self.special_tokens]
        self.itos = self.special_tokens + words
        self.stoi = {token: i for i, token in enumerate(self.itos)}
        self.lowercase = lowercase
        self._unk = self.stoi.get("<unk>")

    @classmethod
    def from_text(
        cls,
        text: str,
        special_tokens: Sequence[str] = (),
        min_freq: int = 1,
        lowercase: bool = True,
    ) -> "WordTokenizer":
        tokens = cls._split(text, lowercase)
        counts = Counter(tokens)
        vocab = sorted(tok for tok, n in counts.items() if n >= min_freq)
        return cls(vocab, special_tokens, lowercase)

    @staticmethod
    def _split(text: str, lowercase: bool = True) -> list[str]:
        if lowercase:
            text = text.lower()
        # \w+ keeps words (and digits) together; everything else is a single token.
        return re.findall(r"\w+|[^\w\s]", text)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int | None:
        return self.stoi.get("<pad>")

    @property
    def bos_id(self) -> int | None:
        return self.stoi.get("<bos>")

    @property
    def eos_id(self) -> int | None:
        return self.stoi.get("<eos>")

    @property
    def unk_id(self) -> int | None:
        return self._unk

    def encode(self, text: str) -> list[int]:
        ids = []
        for token in self._split(text, self.lowercase):
            idx = self.stoi.get(token, self._unk)
            if idx is None:
                raise ValueError(f"unknown token {token!r}; use <unk> or extend the vocab")
            ids.append(idx)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        return " ".join(self.itos[i] for i in ids)

    def _state(self) -> dict:
        base = [t for t in self.itos if t not in self.special_tokens]
        return {
            "type": "word",
            "vocab": base,
            "special_tokens": self.special_tokens,
            "lowercase": self.lowercase,
        }


class BPETokenizer:
    """Byte-pair encoding (Sennrich et al., 2016) trained from scratch.

    Algorithm, in the order it runs:

    1. Split the corpus into words on whitespace. Append ``</w>`` to each word
       so the model learns where words end.
    2. Count every adjacent symbol pair across all words.
    3. Merge the most frequent pair into a new symbol; repeat until the
       requested vocabulary size is reached.
    4. Encoding a new word means replaying the merges in the order they were
       learned (lowest rank first).

    This is the word-level variant used in the original paper and in GPT-2's
    educational reimplementations; it is enough to understand real BPE while
    staying short enough to read in one sitting.
    """

    def __init__(
        self,
        vocab: Sequence[str],
        merges: Sequence[tuple[str, str]],
        special_tokens: Sequence[str] = (),
    ):
        self.special_tokens = list(special_tokens)
        self.itos = self.special_tokens + [t for t in vocab if t not in self.special_tokens]
        self.stoi = {token: i for i, token in enumerate(self.itos)}
        self.merges = [tuple(m) for m in merges]
        self.rank = {pair: i for i, pair in enumerate(self.merges)}
        self._unk = self.stoi.get("<unk>")

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int = 512,
        special_tokens: Sequence[str] = (),
        max_words: int | None = None,
        verbose: bool = False,
    ) -> "BPETokenizer":
        """Learn merges from ``text`` until the vocabulary reaches ``vocab_size``.

        ``max_words`` subsamples the most frequent word types, which keeps
        training on a 1 MB corpus tractable in pure Python.
        """
        counts = Counter(_WORD_RE.findall(text))
        if max_words is not None:
            counts = Counter(dict(counts.most_common(max_words)))
        words = {tuple(word) + (END_OF_WORD,): freq for word, freq in counts.items()}

        base = sorted({symbol for word in words for symbol in word})
        merges: list[tuple[str, str]] = []
        budget = max(vocab_size - len(base) - len(special_tokens), 0)

        while len(merges) < budget:
            pair_counts: Counter[tuple[str, str]] = Counter()
            for word, freq in words.items():
                for pair in zip(word, word[1:]):
                    pair_counts[pair] += freq
            if not pair_counts:
                break
            (best_a, best_b), best_freq = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
            if best_freq < 2:  # nothing is worth merging anymore
                break

            new_words = {}
            for word, freq in words.items():
                merged = _merge_pair(word, (best_a, best_b))
                new_words[merged] = new_words.get(merged, 0) + freq
            words = new_words
            merges.append((best_a, best_b))
            if verbose and len(merges) % 50 == 0:
                print(f"  merges: {len(merges)}  latest: {best_a + best_b!r}")

        merged_symbols = [a + b for a, b in merges]
        vocab = base + merged_symbols
        return cls(vocab, merges, special_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int | None:
        return self.stoi.get("<pad>")

    @property
    def bos_id(self) -> int | None:
        return self.stoi.get("<bos>")

    @property
    def eos_id(self) -> int | None:
        return self.stoi.get("<eos>")

    @property
    def unk_id(self) -> int | None:
        return self._unk

    def _encode_word(self, word: str) -> list[str]:
        symbols = tuple(word) + (END_OF_WORD,)
        while len(symbols) > 1:
            best_rank, best_pos, best_pair = None, None, None
            for pos, pair in enumerate(zip(symbols, symbols[1:])):
                rank = self.rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pos, best_pair = rank, pos, pair
            if best_pair is None:
                break
            symbols = symbols[:best_pos] + (best_pair[0] + best_pair[1],) + symbols[best_pos + 2 :]
        return list(symbols)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in _WORD_RE.findall(text):
            for symbol in self._encode_word(word):
                idx = self.stoi.get(symbol)
                if idx is None:
                    if self._unk is None:
                        raise ValueError(f"unknown symbol {symbol!r}; retrain or add <unk>")
                    idx = self._unk
                ids.append(idx)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        pieces: list[str] = []
        for i in ids:
            token = self.itos[i]
            if token.endswith(END_OF_WORD):
                pieces.append(token[: -len(END_OF_WORD)] + " ")
            else:
                pieces.append(token)
        return "".join(pieces).rstrip()

    def _state(self) -> dict:
        base = [t for t in self.itos if t not in self.special_tokens]
        return {
            "type": "bpe",
            "vocab": base,
            "merges": [list(m) for m in self.merges],
            "special_tokens": self.special_tokens,
        }


def _merge_pair(word: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    """Replace every non-overlapping occurrence of ``pair`` inside ``word``."""
    merged: list[str] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            merged.append(word[i] + word[i + 1])
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)


def save_tokenizer(tokenizer, path: str | Path) -> None:
    """Serialize any tokenizer above to JSON."""
    Path(path).write_text(json.dumps(tokenizer._state(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_tokenizer(path: str | Path):
    """Inverse of :func:`save_tokenizer`."""
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    kind = state.pop("type")
    if kind == "char":
        return CharTokenizer(**state)
    if kind == "word":
        return WordTokenizer(**state)
    if kind == "bpe":
        return BPETokenizer(**state)
    raise ValueError(f"unknown tokenizer type {kind!r}")
