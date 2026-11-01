from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from laboratory.optimization.optimizers import make_optimizer


@pytest.mark.parametrize(
    "name, expected_type",
    [("adamw", torch.optim.AdamW), ("adam", torch.optim.Adam), ("sgd", torch.optim.SGD)],
)
def test_make_optimizer_chooses_the_named_optimizer(name, expected_type):
    parameter = nn.Parameter(torch.zeros(1))
    optimizer = make_optimizer(
        [parameter], name=name, learning_rate=0.01, weight_decay=0.1
    )
    assert isinstance(optimizer, expected_type)
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert optimizer.param_groups[0]["weight_decay"] == 0.1


def test_make_optimizer_rejects_unknown_names():
    parameter = nn.Parameter(torch.zeros(1))
    with pytest.raises(ValueError, match="unknown optimizer"):
        make_optimizer([parameter], name="lbfgs")
