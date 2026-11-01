"""Datasets: indexable collections, one module per task shape.

* :mod:`laboratory.datasets.language` — next-token windows and corpus splits.
* :mod:`laboratory.datasets.sequence` — aligned sequence pairs.

Datasets answer "what is one example"; turning examples into padded tensors is
:mod:`laboratory.preprocessing`. Add ``vision.py`` for image datasets when an
experiment needs it.
"""

from .language import LanguageModelDataset, train_val_split
from .sequence import Seq2SeqDataset

__all__ = ["LanguageModelDataset", "Seq2SeqDataset", "train_val_split"]
