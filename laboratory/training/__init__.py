"""Training loop, schedules, metrics and checkpoints.

* :mod:`laboratory.training.trainer` — the task-agnostic loop and checkpoints.
* :mod:`laboratory.training.schedules` — warmup/cosine and Noam.
* :mod:`laboratory.training.metrics` — running averages and token accuracy.

Training *regimes* that change the loop itself (masked modelling, EMA target
encoders for JEPA, reinforcement learning) belong in the experiment that needs
them first. Move the reusable part here once a second experiment needs it.
"""

from .metrics import RunningMean, accuracy
from .schedules import cosine_warmup, noam
from .trainer import Trainer, load_checkpoint, save_checkpoint

__all__ = [
    "RunningMean",
    "Trainer",
    "accuracy",
    "cosine_warmup",
    "load_checkpoint",
    "noam",
    "save_checkpoint",
]
