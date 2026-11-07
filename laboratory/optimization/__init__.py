"""Optimization: optimizers and learning-rate schedules.

Both are selected by name from the experiment's preset; the default optimizer is
Muon, with AdamW for embeddings, output layers, biases and gains. Schedules are
``torch.optim.lr_scheduler.LambdaLR`` objects, stepped once per optimizer step by
:class:`laboratory.training.Trainer`.
"""

from .muon import Muon, muon_parameter_groups, newton_schulz
from .optimizers import make_optimizer
from .schedules import cosine_warmup, noam

__all__ = ["Muon", "cosine_warmup", "make_optimizer", "muon_parameter_groups", "newton_schulz", "noam"]
