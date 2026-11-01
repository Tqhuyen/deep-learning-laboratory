from __future__ import annotations

import torch
import torch.nn as nn

from laboratory.recurrent import GRU, LSTM, RNN, GRUCell, LSTMCell, RNNCell


def _copy_cell_weights(cell, reference, layer: int = 0) -> None:
    """Copy a PyTorch fused RNN layer into one of our cells."""
    with torch.no_grad():
        cell.linear_ih.weight.copy_(getattr(reference, f"weight_ih_l{layer}"))
        cell.linear_hh.weight.copy_(getattr(reference, f"weight_hh_l{layer}"))
        cell.linear_ih.bias.copy_(getattr(reference, f"bias_ih_l{layer}"))
        cell.linear_hh.bias.copy_(getattr(reference, f"bias_hh_l{layer}"))


def test_lstm_matches_torch_over_a_sequence():
    input_size, hidden_size, steps = 5, 7, 6
    reference = nn.LSTM(input_size, hidden_size, batch_first=True)
    cell = LSTMCell(input_size, hidden_size)
    _copy_cell_weights(cell, reference)

    x = torch.randn(3, steps, input_size)
    h = torch.randn(3, hidden_size)
    c = torch.randn(3, hidden_size)
    expected, (expected_h, expected_c) = reference(x, (h[None], c[None]))

    outputs = []
    for t in range(steps):
        h, (_, c) = cell(x[:, t], (h, c))
        outputs.append(h)

    assert torch.allclose(torch.stack(outputs, dim=1), expected, atol=1e-6)
    assert torch.allclose(h, expected_h[0], atol=1e-6)
    assert torch.allclose(c, expected_c[0], atol=1e-6)


def test_gru_matches_torch_over_a_sequence():
    input_size, hidden_size, steps = 4, 6, 5
    reference = nn.GRU(input_size, hidden_size, batch_first=True)
    cell = GRUCell(input_size, hidden_size)
    _copy_cell_weights(cell, reference)

    x = torch.randn(2, steps, input_size)
    h = torch.randn(2, hidden_size)
    expected, expected_h = reference(x, h[None])

    outputs = []
    for t in range(steps):
        h = cell(x[:, t], h)
        outputs.append(h)

    assert torch.allclose(torch.stack(outputs, dim=1), expected, atol=1e-6)
    assert torch.allclose(h, expected_h[0], atol=1e-6)


def test_vanilla_rnn_matches_torch_over_a_sequence():
    input_size, hidden_size, steps = 4, 5, 5
    reference = nn.RNN(input_size, hidden_size, batch_first=True)
    cell = RNNCell(input_size, hidden_size)
    _copy_cell_weights(cell, reference)

    x = torch.randn(2, steps, input_size)
    h = torch.randn(2, hidden_size)
    expected, expected_h = reference(x, h[None])

    outputs = []
    for t in range(steps):
        h = cell(x[:, t], h)
        outputs.append(h)

    assert torch.allclose(torch.stack(outputs, dim=1), expected, atol=1e-6)
    assert torch.allclose(h, expected_h[0], atol=1e-6)


def test_stacked_lstm_matches_torch():
    input_size, hidden_size, num_layers = 3, 4, 2
    reference = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
    model = LSTM(input_size, hidden_size, num_layers=num_layers)
    for layer in range(num_layers):
        _copy_cell_weights(model.cells[layer], reference, layer)

    model.eval()
    x = torch.randn(2, 5, input_size)
    output, (h_n, c_n) = model(x)
    expected_output, (expected_h, expected_c) = reference(x)

    assert torch.allclose(output, expected_output, atol=1e-6)
    assert torch.allclose(h_n, expected_h, atol=1e-6)
    assert torch.allclose(c_n, expected_c, atol=1e-6)


def test_recurrent_wrappers_return_documented_shapes():
    x = torch.randn(2, 5, 4)
    for network in (RNN(4, 8, num_layers=2), GRU(4, 8, num_layers=2), LSTM(4, 8, num_layers=2)):
        output, state = network(x)
        assert output.shape == (2, 5, 8)
