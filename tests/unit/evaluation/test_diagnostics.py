from __future__ import annotations

import math

import pytest
import torch

from laboratory.evaluation.diagnostics import (
    decay_slope,
    retention,
    timestep_gradient_norms,
)
from laboratory.models.recurrent import GRU, LSTM, RNN


def _linear_rnn(size: int, recurrent_scale: float) -> RNN:
    """Single-layer RNN with W_ih = I, W_hh = a I and zero biases."""
    network = RNN(size, size)
    cell = network.cells[0]
    with torch.no_grad():
        cell.linear_ih.weight.copy_(torch.eye(size))
        cell.linear_hh.weight.copy_(recurrent_scale * torch.eye(size))
        cell.linear_ih.bias.zero_()
        cell.linear_hh.bias.zero_()
    return network


def test_reach_matches_the_analytic_decay_rate_of_a_linear_rnn():
    # With tiny inputs tanh is the identity, so h_T = sum_t a^(T-t) x_t and
    # || d||h_T||^2 / d x_t || = 2 ||h_T|| a^(T-t): log-slope over lag = log a.
    scale = 0.5
    network = _linear_rnn(4, scale)
    inputs = 1e-4 * torch.randn(3, 12, 4)
    norms = timestep_gradient_norms(network, inputs)
    assert decay_slope(norms) == pytest.approx(math.log(scale), abs=1e-3)
    assert retention(norms, [1, 2])[2] == pytest.approx(scale**2, rel=1e-3)


def test_reach_reaches_earlier_timesteps_through_the_recurrence():
    # Regression: differentiating the stacked outputs gave zeros before step T.
    network = _linear_rnn(3, 0.9)
    norms = timestep_gradient_norms(network, 1e-3 * torch.randn(2, 6, 3))
    assert (norms[:-1] > 0).all()


def test_reach_is_zero_before_the_last_step_without_recurrence():
    network = _linear_rnn(3, 0.0)
    norms = timestep_gradient_norms(network, torch.randn(2, 6, 3))
    assert torch.all(norms[:-1] == 0)
    assert norms[-1] > 0


def _gated_lstm(size: int, forget_bias: float) -> LSTM:
    """LSTM whose candidate is the input, gates saturated by their biases."""
    network = LSTM(size, size)
    cell = network.cells[0]
    with torch.no_grad():
        cell.linear_ih.weight.zero_()
        cell.linear_hh.weight.zero_()
        cell.linear_hh.bias.zero_()
        cell.linear_ih.weight[2 * size : 3 * size].copy_(torch.eye(size))  # g = tanh(x)
        bias = torch.zeros(4 * size)
        bias[:size] = 20.0  # input gate open
        bias[size : 2 * size] = forget_bias
        bias[3 * size :] = 20.0  # output gate open
        cell.linear_ih.bias.copy_(bias)
    return network


def test_open_forget_gate_keeps_reach_and_closed_gate_cuts_it():
    inputs = 1e-3 * torch.randn(2, 10, 3)
    carry = timestep_gradient_norms(_gated_lstm(3, forget_bias=20.0), inputs)
    forget = timestep_gradient_norms(_gated_lstm(3, forget_bias=-20.0), inputs)
    assert decay_slope(carry) == pytest.approx(0.0, abs=1e-3)
    assert decay_slope(forget) < -10.0


def test_reach_probe_runs_for_every_cell_and_leaves_parameter_gradients_alone():
    for network in (RNN(6, 8), GRU(6, 8), LSTM(6, 8)):
        norms = timestep_gradient_norms(network, torch.randn(2, 9, 6))
        assert norms.shape == (9,)
        assert torch.isfinite(norms).all()
        assert all(parameter.grad is None for parameter in network.parameters())


def test_decay_slope_stops_fitting_at_the_numerical_floor():
    # Lag order is reversed internally: the last element is lag 0.
    norms = [0.0, 0.0, 1e-3, 1e-2, 1e-1, 1.0]
    assert decay_slope(norms) == pytest.approx(math.log(0.1), abs=1e-9)
    assert math.isnan(decay_slope([1.0]))


def test_fixed_length_sequences_keeps_only_long_enough_rows_without_padding():
    from laboratory.evaluation.diagnostics import fixed_length_sequences

    rows = fixed_length_sequences([[1, 2, 3, 4], [5, 6], [7, 8, 9]], length=3, limit=5)
    assert rows.tolist() == [[1, 2, 3], [7, 8, 9]]
    assert fixed_length_sequences([[1, 2, 3], [4, 5, 6]], length=2, limit=1).tolist() == [[1, 2]]
    with pytest.raises(ValueError, match="at least 4"):
        fixed_length_sequences([[1, 2]], length=4)
