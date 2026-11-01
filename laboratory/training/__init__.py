"""Training: the loop and checkpointing.

* :mod:`laboratory.training.trainer` — task-agnostic loop.
* :mod:`laboratory.training.checkpoints` — save and load weights plus metadata.

Optimisers and schedules are chosen in :mod:`laboratory.optimization`; metrics
live in :mod:`laboratory.evaluation`. Training regimes that change the loop
itself (EMA target encoders for JEPA, reinforcement learning) start in the
experiment that needs them.
"""

from .checkpoints import load_checkpoint, save_checkpoint
from .trainer import Trainer

__all__ = ["Trainer", "load_checkpoint", "save_checkpoint"]
