from __future__ import annotations

import torch
import torch.nn as nn

from laboratory.evaluation.landscape import (
    filter_normalized_direction,
    loss_surface,
    orthogonalize,
    shifted_state_dict,
)


def test_direction_preserves_each_filter_norm():
    base = {
        "weight": torch.randn(4, 3),
        "bias": torch.randn(4),
    }
    generator = torch.Generator().manual_seed(0)
    direction = filter_normalized_direction(base, generator)
    assert direction["weight"].shape == base["weight"].shape
    assert torch.allclose(
        direction["weight"].norm(dim=1), base["weight"].norm(dim=1), atol=1e-6
    )
    assert torch.allclose(direction["bias"].norm(), base["bias"].norm(), atol=1e-6)


def test_orthogonalized_direction_has_no_component_along_the_first():
    first = {"weight": torch.randn(3, 5), "bias": torch.randn(3)}
    second = {"weight": torch.randn(3, 5), "bias": torch.randn(3)}
    orthogonal = orthogonalize(first, second)
    for name in first:
        assert abs(torch.dot(orthogonal[name].reshape(-1), first[name].reshape(-1))) < 1e-5


def test_shifted_state_dict_moves_parameters_and_keeps_buffers():
    base = {"weight": torch.zeros(2, 2), "step": torch.tensor(7)}
    first = {"weight": torch.ones(2, 2)}
    second = {"weight": torch.full((2, 2), 2.0)}
    shifted = shifted_state_dict(base, first, second, 0.5, -1.0)
    assert torch.allclose(shifted["weight"], torch.full((2, 2), -1.5))
    assert shifted["step"].item() == 7


def test_loss_surface_restores_the_base_state_and_is_minimal_at_the_centre():
    model = nn.Linear(3, 1)
    true_weight = torch.tensor([[1.0], [2.0], [-1.0]])
    with torch.no_grad():
        model.weight.copy_(true_weight.T)
        model.bias.fill_(0.2)
    base = {name: value.detach().clone() for name, value in model.state_dict().items()}
    generator = torch.Generator().manual_seed(0)
    first = filter_normalized_direction(base, generator)
    second = orthogonalize(first, filter_normalized_direction(base, generator))

    inputs = torch.randn(16, 3)
    targets = inputs @ true_weight + 0.2
    batches = [(inputs, targets)]

    def step(current_model, batch):
        return torch.nn.functional.mse_loss(current_model(batch[0]), batch[1])

    grid = [-1.0, 0.0, 1.0]
    surface = loss_surface(model, base, first, second, grid, batches, step)
    assert surface.shape == (3, 3)
    assert surface[1, 1] <= surface.min() + 1e-9
    for name, value in model.state_dict().items():
        assert torch.allclose(value, base[name])
