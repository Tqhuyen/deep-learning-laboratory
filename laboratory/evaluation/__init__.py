"""Evaluation: metrics and task-level checks.

Metrics live here so an experiment's final evaluation does not depend on the
training loop. Add task-specific evaluators (BLEU, retrieval recall) as their own
module once a second experiment needs them.
"""

from .diagnostics import decay_slope, timestep_gradient_norms
from .landscape import (
    filter_normalized_direction,
    loss_surface,
    orthogonalize,
    shifted_state_dict,
)
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
    "filter_normalized_direction",
    "holm_adjust",
    "loss_surface",
    "mcnemar_exact",
    "orthogonalize",
    "paired_bootstrap_difference",
    "paired_bootstrap_p_value",
    "perplexity",
    "shifted_state_dict",
    "timestep_gradient_norms",
]
