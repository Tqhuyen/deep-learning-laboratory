"""Loss-landscape slices around a trained checkpoint.

A 2-D slice is the surface

$$
\\theta(\\alpha,\\beta)=\\theta_0+\\alpha\\delta_1+\\beta\\delta_2
$$

where ``\\delta_1`` and ``\\delta_2`` are filter-normalised random directions: each
row (filter) of a matrix gets a random direction with the same norm as the
corresponding trained row, so the plot is not distorted by the scale symmetry of
the network. The two directions are orthogonalised before use.

This is a local diagnostic. It can show that a minimum is flat or sharp, or that
a barrier separates two points; it cannot establish generalisation or a universal
ordering between architectures. Those require a statement over a task class.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
import torch


def filter_normalized_direction(
    state_dict: Mapping[str, torch.Tensor],
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Random direction whose per-row norm equals the trained tensor's row norm."""
    direction: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if not tensor.is_floating_point():
            continue
        noise = torch.randn(
            tensor.shape,
            generator=generator,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        if tensor.dim() >= 2:
            rows = tensor.reshape(tensor.shape[0], -1)
            noise_rows = noise.reshape(tensor.shape[0], -1)
            noise_rows = noise_rows / noise_rows.norm(dim=1, keepdim=True).clamp_min(1e-12)
            noise_rows = noise_rows * rows.norm(dim=1, keepdim=True).clamp_min(1e-12)
            direction[name] = noise_rows.reshape(tensor.shape)
        else:
            direction[name] = (
                noise / noise.norm().clamp_min(1e-12) * tensor.norm().clamp_min(1e-12)
            )
    return direction


def orthogonalize(
    first: Mapping[str, torch.Tensor],
    second: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remove the component of ``second`` along ``first``, per tensor."""
    result: dict[str, torch.Tensor] = {}
    for name, tensor in second.items():
        reference = first.get(name)
        if reference is None or reference.numel() == 0:
            result[name] = tensor
            continue
        flat_reference = reference.reshape(-1)
        flat_tensor = tensor.reshape(-1)
        projection = (flat_tensor @ flat_reference) / flat_reference.pow(2).sum().clamp_min(1e-12)
        flat_tensor = flat_tensor - projection * flat_reference
        # Preserve the original norm scale after changing direction.
        flat_tensor = flat_tensor / flat_tensor.norm().clamp_min(1e-12) * tensor.reshape(-1).norm().clamp_min(1e-12)
        result[name] = flat_tensor.reshape(tensor.shape)
    return result


def shifted_state_dict(
    base: Mapping[str, torch.Tensor],
    first: Mapping[str, torch.Tensor],
    second: Mapping[str, torch.Tensor],
    alpha: float,
    beta: float,
) -> dict[str, torch.Tensor]:
    """``base + alpha * first + beta * second``, tensor by tensor."""
    return {
        name: (
            base[name] + alpha * first[name] + beta * second[name]
            if name in first
            else base[name]
        )
        for name in base
    }


def loss_surface(
    model: torch.nn.Module,
    base_state: Mapping[str, torch.Tensor],
    first: Mapping[str, torch.Tensor],
    second: Mapping[str, torch.Tensor],
    grid: Sequence[float],
    batches: Sequence[Mapping[str, torch.Tensor]] | Sequence[tuple[torch.Tensor, ...]],
    step_fn: Callable,
) -> np.ndarray:
    """Evaluate the mean loss on ``grid x grid``; leaves the model at ``base_state``.

    ``step_fn(model, batch)`` must return a scalar tensor, exactly like the
    training step. Batches must already live on the model's device.
    """
    surface = np.zeros((len(grid), len(grid)), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for row, alpha in enumerate(grid):
            for column, beta in enumerate(grid):
                model.load_state_dict(shifted_state_dict(base_state, first, second, alpha, beta))
                total = 0.0
                for batch in batches:
                    loss = step_fn(model, batch)
                    total += float(loss)
                surface[row, column] = total / max(len(batches), 1)
    model.load_state_dict(base_state)
    return surface
