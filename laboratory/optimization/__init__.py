"""Optimization: optimizers and learning-rate schedules.

Both are selected by name from the experiment's preset. Schedules are
``torch.optim.lr_scheduler.LambdaLR`` objects, stepped once per optimizer step by
:class:`laboratory.training.Trainer`.
"""

from .optimizers import make_optimizer
from .schedules import cosine_warmup, noam

__all__ = ["cosine_warmup", "make_optimizer", "noam"]
