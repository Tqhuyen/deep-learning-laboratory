"""Diagnostics that expose optimisation mechanics.

The gradient-reach probe answers one question: how strongly does the final
hidden state depend on the input ``lag`` steps earlier? For a vanilla RNN

$$
\\frac{\\partial h_T}{\\partial x_t}
= \\Bigl(\\prod_{k=t+1}^{T} \\frac{\\partial h_k}{\\partial h_{k-1}}\\Bigr)
  \\frac{\\partial h_t}{\\partial x_t},
$$

so the norm decays roughly geometrically with the lag when the recurrent
Jacobians are contractive. An LSTM adds the cell-state path
``c_T = f_T * c_{T-1} + ...``, which preserves sensitivity while the forget
gates stay near one.

The numbers are measurements on a trained network: they can falsify a
mechanistic explanation, they cannot prove generalisation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def timestep_gradient_norms(recurrent: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Mean ``|| d ||h_T||^2 / d x_t ||`` over the batch, one value per timestep.

    ``inputs`` are embedded, **unpadded** sequences ``(batch, time, input_size)``;
    ``h_T`` is the top-layer state after the last timestep. The derivative is
    taken with respect to the inputs, so the only path from ``x_t`` to ``h_T``
    is the recurrence itself. (Differentiating with respect to the stacked
    output tensor would be wrong: the stack is a copy whose earlier slices are
    not connected to ``h_T``, so every gradient except the last would be zero.)

    Parameter ``.grad`` fields are not touched.
    """
    inputs = inputs.detach().requires_grad_(True)
    outputs, _ = recurrent(inputs)
    loss = outputs[:, -1].pow(2).sum()
    (gradients,) = torch.autograd.grad(loss, inputs)
    return gradients.norm(dim=-1).mean(dim=0).detach().cpu()


def fixed_length_sequences(
    sequences: Sequence[Sequence[int]],
    length: int,
    limit: int | None = None,
) -> torch.Tensor:
    """First ``length`` ids of every sequence with at least ``length`` ids, stacked.

    The reach probe needs unpadded inputs: with right padding, the final
    timestep of a short row is a pad, not the row's last real token, and the
    measured "final state" would be the state after the pads.
    """
    rows = [list(sequence[:length]) for sequence in sequences if len(sequence) >= length]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"no sequence has at least {length} tokens")
    return torch.tensor(rows, dtype=torch.long)


def decay_slope(norms: Sequence[float], relative_floor: float = 1e-10) -> float:
    """Least-squares slope of ``log(norm)`` against lag (lag 0 = final timestep).

    ``norms`` are ordered by timestep, as returned by
    :func:`timestep_gradient_norms`; they are reversed so that position ``k`` is
    lag ``k``. The fit stops at the first lag whose norm falls below
    ``relative_floor`` times the largest norm, because beyond that point the
    values are floating-point residue rather than signal.

    ``exp(slope)`` is the average per-step retention factor; ``ln 2 / |slope|``
    is the half-life in steps. Returns ``nan`` when fewer than two lags remain.
    """
    values = np.asarray(norms, dtype=float)[::-1]
    if values.size < 2 or not np.isfinite(values).all() or values.max() <= 0.0:
        return float("nan")
    above = values > values.max() * relative_floor
    if not above.all():
        values = values[: int(np.argmin(above))]
    if values.size < 2:
        return float("nan")
    return float(np.polyfit(np.arange(values.size, dtype=float), np.log(values), 1)[0])


def retention(norms: Sequence[float], lags: Sequence[int]) -> dict[int, float]:
    """Gradient norm at each lag relative to lag 0 (``g(k) / g(0)``)."""
    values = np.asarray(norms, dtype=float)[::-1]
    if values.size == 0 or values[0] <= 0.0:
        return {int(lag): float("nan") for lag in lags}
    return {
        int(lag): float(values[lag] / values[0]) if lag < values.size else float("nan")
        for lag in lags
    }
