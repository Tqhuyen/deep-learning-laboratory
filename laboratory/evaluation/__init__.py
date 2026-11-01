"""Evaluation: metrics and task-level checks.

Metrics live here so an experiment's final evaluation does not depend on the
training loop. Add task-specific evaluators (BLEU, retrieval recall) as their own
module once a second experiment needs them.
"""

from .metrics import accuracy, exact_match, perplexity

__all__ = ["accuracy", "exact_match", "perplexity"]
