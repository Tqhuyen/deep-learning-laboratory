"""Small helpers shared by every part of the library.

Nothing here is deep-learning specific except the mask builders. The mask
builders define the *convention used everywhere in this repo*:

    A boolean mask is **True where attention is allowed** and False where the
    position must be hidden. Scores are filled with -inf where the mask is
    False, so softmax gives them probability 0.

PyTorch's ``F.scaled_dot_product_attention`` uses the same convention, which
keeps this code portable to the fused kernel later.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 0) -> None:
    """Seed python, numpy and torch (CPU + CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "auto") -> torch.device:
    """Return ``prefer`` if available, else CPU.

    ``"auto"`` means "CUDA when it exists, otherwise CPU", which is what every
    experiment uses. An explicit CUDA request also falls back to CPU when CUDA
    is unavailable, so presets stay runnable everywhere.
    """
    if prefer == "auto":
        prefer = "cuda"
    if prefer.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(prefer)


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Number of parameters, e.g. ``count_parameters(model) / 1e6`` for millions."""
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def causal_mask(seq_len: int, device: torch.device | str | None = None) -> torch.Tensor:
    """Lower-triangular boolean mask, shape ``(seq_len, seq_len)``.

    ``mask[i, j]`` is True when query ``i`` may see key ``j`` (``j <= i``).
    This is what makes a decoder *autoregressive*: token ``i`` cannot look at
    its own future.
    """
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def lengths_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """Padding mask, shape ``(batch, max_len)``, True at real tokens.

    ``lengths`` is the number of *real* tokens per row (before padding).
    """
    if max_len is None:
        max_len = int(lengths.max())
    positions = torch.arange(max_len, device=lengths.device)
    return positions[None, :] < lengths[:, None]
