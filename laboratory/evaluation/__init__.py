"""Evaluation: metrics and task-level checks.

Metrics live here so an experiment's final evaluation does not depend on the
training loop. Add task-specific evaluators (BLEU, retrieval recall) as their own
module once a second experiment needs them.
"""

from .diagnostics import decay_slope, timestep_gradient_norms
from .metrics import accuracy, exact_match, perplexity
from .statistics import (
    bootstrap_confidence_interval,
    holm_adjust,
    mcnemar_exact,
    paired_bootstrap_difference,
    paired_bootstrap_p_value,
)

__all__ = [
    "accuracy",
    "bootstrap_confidence_interval",
    "decay_slope",
    "exact_match",
    "holm_adjust",
    "mcnemar_exact",
    "paired_bootstrap_difference",
    "paired_bootstrap_p_value",
    "perplexity",
    "timestep_gradient_norms",
]
