"""Preprocessing: text to ids and examples to padded tensors.

* :mod:`laboratory.preprocessing.text` — tokenizers (character, word, byte-pair).
* :mod:`laboratory.preprocessing.batching` — padding, collators, random batches.

Add ``vision.py`` (decode, resize, patchify, augment) or ``audio.py`` when a
modality exists. Preprocessing must stay deterministic for a given seed so an
experiment can be reproduced on another machine.
"""

from .batching import Seq2SeqCollator, pad_sequences, random_lm_batch
from .text import (
    BPETokenizer,
    CharTokenizer,
    WordTokenizer,
    load_tokenizer,
    save_tokenizer,
)

__all__ = [
    "BPETokenizer",
    "CharTokenizer",
    "Seq2SeqCollator",
    "WordTokenizer",
    "load_tokenizer",
    "pad_sequences",
    "random_lm_batch",
    "save_tokenizer",
]
