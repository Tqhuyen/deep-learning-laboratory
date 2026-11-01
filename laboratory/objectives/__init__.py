"""Training objectives (losses).

One module per task family. Every objective takes ``(predictions, targets)``
and returns a scalar tensor, so an experiment can swap losses without touching
the training loop:

* :mod:`laboratory.objectives.language` — next-token and sequence losses, with
  padding handled by ``ignore_index``.

Add ``regression.py`` for latent-prediction objectives (JEPA-style), or
``contrastive.py`` for InfoNCE, when an experiment needs them. Do not add an
objective that no experiment runs.
"""

from .language import token_cross_entropy

__all__ = ["token_cross_entropy"]
