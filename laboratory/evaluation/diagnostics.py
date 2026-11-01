"""Diagnostics that expose optimisation mechanics.

The gradient-decay probe answers a specific question: how much does the final
hidden state depend on an input ``lag`` steps earlier? For a vanilla RNN the
recurrence Jacobian is a product of state-to-state matrices, so the norm decays
like a power of the largest singular value; gated cells keep a more direct path
through the cell state. The numbers here are measurements, not a proof, but they
let a theoretical bound be checked against the trained network.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def timestep_gradient_norms(recurrent: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Mean ``||dL/dh_t||`` for ``L = ||h_last||^2``, one value per timestep.

    Uses ``torch.autograd.grad`` with respect to the stack output, so parameter
    gradients are not touched. Inputs are the embedded sequence, shape
    ``(batch, time, input_size)``.
    """
    outputs, _ = recurrent(inputs)
    loss = outputs[:, -1].pow(2).sum()
    (gradients,) = torch.autograd.grad(loss, outputs)
    return gradients.norm(dim=-1).mean(dim=0).detach().cpu()


def decay_slope(norms: Sequence[float]) -> float:
    """Slope of ``log(norm)`` against lag from the final step; more negative is faster decay."""
    values = np.log(np.asarray(norms, dtype=float)[::-1] + 1e-12)
    positions = np.arange(values.size, dtype=float)
    if values.size < 2:
        return 0.0
    return float(np.polyfit(positions, values, 1)[0])
