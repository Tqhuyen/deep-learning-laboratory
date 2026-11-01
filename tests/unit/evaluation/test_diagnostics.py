from __future__ import annotations

import torch

from laboratory.evaluation.diagnostics import decay_slope, timestep_gradient_norms
from laboratory.models.recurrent import GRU, LSTM, RNN


def test_gradient_probe_returns_one_finite_value_per_timestep():
    for network in (RNN(6, 8), GRU(6, 8), LSTM(6, 8)):
        inputs = torch.randn(2, 9, 6)
        norms = timestep_gradient_norms(network, inputs)
        assert norms.shape == (9,)
        assert torch.isfinite(norms).all()
        assert (norms >= 0).all()


def test_gradient_probe_does_not_touch_parameter_gradients():
    network = RNN(4, 5)
    inputs = torch.randn(1, 6, 4)
    timestep_gradient_norms(network, inputs)
    assert all(parameter.grad is None for parameter in network.parameters())


def test_decay_slope_is_negative_when_reach_shrinks_with_lag():
    increasing_in_time = [0.01, 0.1, 1.0]
    assert decay_slope(increasing_in_time) < 0.0
    assert abs(decay_slope([1.0, 1.0, 1.0])) < 1e-6
