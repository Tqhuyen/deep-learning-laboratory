"""Inference: generation and decoding.

* :func:`laboratory.inference.generate` — decode-only models.
* :func:`laboratory.inference.greedy_decode` — encoder-decoder models.

Inference never changes parameters; it puts the model in evaluation mode and
restores the previous mode. Sampling strategy is a parameter of the call, not a
state on the model.
"""

from .generation import generate, next_token
from .sequence_to_sequence import greedy_decode

__all__ = ["generate", "greedy_decode", "next_token"]
