"""Data handling.

* :mod:`laboratory.data.tokenization` — text to ids and back.
* :mod:`laboratory.data.batching` — language-model windows, sequence-to-sequence
  batching with teacher forcing, padding helpers.

Add one module per modality when it is actually needed: for example
``vision.py`` for image datasets and patchification, ``audio.py`` for features.
Keep the batching conventions (right padding, boolean masks True = real token)
identical so the trainer and the blocks keep working.
"""

from .batching import (
    LanguageModelDataset,
    Seq2SeqCollator,
    Seq2SeqDataset,
    pad_sequences,
    random_lm_batch,
    train_val_split,
)
from .tokenization import (
    BPETokenizer,
    CharTokenizer,
    WordTokenizer,
    load_tokenizer,
    save_tokenizer,
)

__all__ = [
    "BPETokenizer",
    "CharTokenizer",
    "LanguageModelDataset",
    "Seq2SeqCollator",
    "Seq2SeqDataset",
    "WordTokenizer",
    "load_tokenizer",
    "pad_sequences",
    "random_lm_batch",
    "save_tokenizer",
    "train_val_split",
]
