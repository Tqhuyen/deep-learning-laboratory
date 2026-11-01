"""Losses.

One module per task family, all with the signature
``(predictions, targets, **options) -> scalar tensor``:

* :mod:`laboratory.losses.language` — token cross-entropy with ``ignore_index``.

Add ``regression.py`` for latent-prediction objectives (JEPA), or
``contrastive.py`` for InfoNCE, when an experiment needs them. Do not add a loss
that no experiment runs.
"""

from .language import token_cross_entropy

__all__ = ["token_cross_entropy"]

from .classification import classification_cross_entropy

__all__ = ["classification_cross_entropy", "token_cross_entropy"]
