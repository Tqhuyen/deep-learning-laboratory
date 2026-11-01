"""Recurrent models, from the equations up.

No ``nn.RNN``, no ``nn.LSTM`` — just ``nn.Linear`` and the gate arithmetic from
the papers, so you can see (and modify) every addition:

* :class:`RNNCell` / :class:`RNN` — Elman (1990).
* :class:`LSTMCell` / :class:`LSTM` — Hochreiter & Schmidhuber (1997).
* :class:`GRUCell` / :class:`GRU` — Cho et al. (2014).

Gate order matches PyTorch's fused kernels (``i, f, g, o`` for LSTM and
``r, z, n`` for GRU). That is deliberate: the tests copy weights into
``nn.LSTM`` / ``nn.GRU`` and check that the outputs match, which is the only
real proof that the equations are right.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..losses.language import token_cross_entropy


class RNNCell(nn.Module):
    r"""$$ h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1}) $$"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.linear_ih = nn.Linear(input_size, hidden_size)
        self.linear_hh = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear_ih(x) + self.linear_hh(h))


class LSTMCell(nn.Module):
    r"""One LSTM step.

    Gate pre-activations are stacked as ``[i, f, g, o]``::

        i = sigmoid(W_i x + U_i h)        input gate
        f = sigmoid(W_f x + U_f h)        forget gate
        g = tanh(W_g x + U_g h)           candidate cell content
        o = sigmoid(W_o x + U_o h)        output gate
        c = f * c_prev + i * g
        h = o * tanh(c)

    The cell state ``c`` is the "conveyor belt": it only undergoes elementwise
    multiplication and addition, which is what lets gradients survive long
    sequences.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.linear_ih = nn.Linear(input_size, 4 * hidden_size)
        self.linear_hh = nn.Linear(hidden_size, 4 * hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        h, c = state
        gates = self.linear_ih(x) + self.linear_hh(h)
        i, f, g, o = gates.chunk(4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, (h, c)


class GRUCell(nn.Module):
    r"""Gated recurrent unit.

    Gate pre-activations are stacked as ``[r, z, n]``::

        r = sigmoid(W_r x + U_r h)                 reset gate
        z = sigmoid(W_z x + U_z h)                 update gate
        n = tanh(W_n x + r * (U_n h))              candidate
        h = (1 - z) * n + z * h
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.linear_ih = nn.Linear(input_size, 3 * hidden_size)
        self.linear_hh = nn.Linear(hidden_size, 3 * hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        gates_i = self.linear_ih(x)
        gates_h = self.linear_hh(h)
        i_r, i_z, i_n = gates_i.chunk(3, dim=-1)
        h_r, h_z, h_n = gates_h.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        return (1 - z) * n + z * h


class RNN(nn.Module):
    """Stacked :class:`RNNCell` applied over the time axis.

    Input is ``(B, T, input_size)``; output is ``(B, T, hidden_size)`` plus the
    final hidden state ``(num_layers, B, hidden_size)``.
    """

    cell_type = RNNCell

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [
                self.cell_type(input_size if layer == 0 else hidden_size, hidden_size)
                for layer in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def init_state(self, batch: int, device, dtype):
        return [torch.zeros(batch, self.hidden_size, device=device, dtype=dtype) for _ in range(self.num_layers)]

    def forward(self, x: torch.Tensor, hidden=None):
        batch, steps, _ = x.shape
        if hidden is None:
            hidden = self.init_state(batch, x.device, x.dtype)

        outputs = []
        for step in range(steps):
            inp = x[:, step]
            new_hidden = []
            for layer, cell in enumerate(self.cells):
                h = cell(inp, hidden[layer])
                new_hidden.append(h)
                inp = self.dropout(h) if layer < self.num_layers - 1 else h
            hidden = new_hidden
            outputs.append(inp)

        return torch.stack(outputs, dim=1), torch.stack(hidden, dim=0)


class LSTM(RNN):
    """Stacked :class:`LSTMCell`; hidden state is a tuple ``(h, c)``."""

    cell_type = LSTMCell

    def init_state(self, batch: int, device, dtype):
        h = super().init_state(batch, device, dtype)
        c = [torch.zeros_like(state) for state in h]
        return list(zip(h, c))

    def forward(self, x: torch.Tensor, hidden=None):
        batch, steps, _ = x.shape
        if hidden is None:
            hidden = self.init_state(batch, x.device, x.dtype)

        outputs = []
        for step in range(steps):
            inp = x[:, step]
            new_hidden = []
            for layer, cell in enumerate(self.cells):
                h, (h_n, c_n) = cell(inp, hidden[layer])
                new_hidden.append((h_n, c_n))
                inp = self.dropout(h) if layer < self.num_layers - 1 else h
            hidden = new_hidden
            outputs.append(inp)

        h_n = torch.stack([state[0] for state in hidden], dim=0)
        c_n = torch.stack([state[1] for state in hidden], dim=0)
        return torch.stack(outputs, dim=1), (h_n, c_n)


class GRU(RNN):
    """Stacked :class:`GRUCell`."""

    cell_type = GRUCell


# Task heads built on the cells above. They live here (not in an experiment)
# because both comparison experiments reuse them and their parameter counts are
# what `laboratory.models.capacity` matches.

RECURRENT_LAYERS = {"rnn": RNN, "lstm": LSTM, "gru": GRU}


class RecurrentLanguageModel(nn.Module):
    """Embedding -> recurrent stack -> next-token logits.

    ``forward`` returns ``(logits, loss)``; ``loss`` is ``None`` without
    targets, matching :class:`laboratory.models.GPT`.
    """

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int = 64,
        hidden_size: int = 256,
        num_layers: int = 1,
        cell: str = "lstm",
        dropout: float = 0.0,
    ):
        super().__init__()
        if cell not in RECURRENT_LAYERS:
            raise ValueError(f"unknown cell {cell!r}; choose from {sorted(RECURRENT_LAYERS)}")
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.recurrent = RECURRENT_LAYERS[cell](
            embedding_dim, hidden_size, num_layers=num_layers, dropout=dropout
        )
        self.head = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor | None = None):
        outputs, _ = self.recurrent(self.embedding(inputs))
        logits = self.head(outputs)
        loss = token_cross_entropy(logits, targets) if targets is not None else None
        return logits, loss


class RecurrentSequenceClassifier(nn.Module):
    """Embedding -> recurrent stack -> last real hidden -> class logits.

    ``lengths`` is required: with right padding, the final hidden state of a row
    is the state after its pads, so the representation must be gathered at
    ``lengths - 1``.
    """

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int = 64,
        hidden_size: int = 256,
        num_layers: int = 1,
        cell: str = "lstm",
        num_classes: int = 2,
        dropout: float = 0.0,
        pad_id: int | None = None,
    ):
        super().__init__()
        if cell not in RECURRENT_LAYERS:
            raise ValueError(f"unknown cell {cell!r}; choose from {sorted(RECURRENT_LAYERS)}")
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=pad_id)
        self.recurrent = RECURRENT_LAYERS[cell](
            embedding_dim, hidden_size, num_layers=num_layers, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.recurrent(self.embedding(inputs))
        last = (lengths - 1).clamp(min=0)
        rows = torch.arange(outputs.size(0), device=outputs.device)
        hidden = outputs[rows, last]
        return self.head(self.dropout(hidden))
