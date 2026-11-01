"""Parameter-count formulas and matched-capacity model sizing.

Comparing RNN, GRU and LSTM at the same number of *hidden units* is not a fair
comparison: one LSTM unit owns four gate matrices, a GRU unit three, an RNN unit
one. This module writes the exact parameter counts of the models in
:mod:`laboratory.models.recurrent` and solves for the hidden size that hits a
shared parameter budget.

For one recurrent layer with input size ``I`` and hidden size ``H``:

* RNN:  ``(I*H + H) + (H*H + H)``
* GRU:  ``3 * ((I*H + H) + (H*H + H))``
* LSTM: ``4 * ((I*H + H) + (H*H + H))``

so the multiplier is 1, 3 or 4. The embedding and the task head are added on
top. The formulas are tested against ``count_parameters`` on real modules.
"""

from __future__ import annotations

from typing import Callable

GATE_MULTIPLIERS = {"rnn": 1, "gru": 3, "lstm": 4}


def recurrent_stack_parameters(
    cell: str,
    input_size: int,
    hidden_size: int,
    num_layers: int = 1,
) -> int:
    """Parameters of the stacked cells, including biases, excluding embeddings."""
    if cell not in GATE_MULTIPLIERS:
        raise ValueError(f"unknown cell {cell!r}; choose from {sorted(GATE_MULTIPLIERS)}")
    multiplier = GATE_MULTIPLIERS[cell]
    total = 0
    for layer in range(num_layers):
        layer_input = input_size if layer == 0 else hidden_size
        total += multiplier * (layer_input * hidden_size + hidden_size * hidden_size + 2 * hidden_size)
    return total


def language_model_parameters(
    cell: str,
    vocabulary_size: int,
    embedding_dim: int,
    hidden_size: int,
    num_layers: int = 1,
) -> int:
    """Exact parameter count of :class:`RecurrentLanguageModel`."""
    return (
        vocabulary_size * embedding_dim
        + recurrent_stack_parameters(cell, embedding_dim, hidden_size, num_layers)
        + hidden_size * vocabulary_size
        + vocabulary_size
    )


def sequence_classifier_parameters(
    cell: str,
    vocabulary_size: int,
    embedding_dim: int,
    hidden_size: int,
    num_layers: int = 1,
    num_classes: int = 2,
) -> int:
    """Exact parameter count of :class:`RecurrentSequenceClassifier`."""
    return (
        vocabulary_size * embedding_dim
        + recurrent_stack_parameters(cell, embedding_dim, hidden_size, num_layers)
        + hidden_size * num_classes
        + num_classes
    )


def _match(counter: Callable[[int], int], target: int, minimum: int, maximum: int) -> tuple[int, int]:
    low, high = minimum, maximum
    while high - low > 1:
        middle = (low + high) // 2
        if counter(middle) <= target:
            low = middle
        else:
            high = middle
    candidates = [(hidden, counter(hidden)) for hidden in (low, high)]
    return min(candidates, key=lambda item: abs(item[1] - target))


def match_language_model_hidden_size(
    cell: str,
    target_parameters: int,
    vocabulary_size: int,
    embedding_dim: int,
    num_layers: int = 1,
    minimum: int = 8,
    maximum: int = 4096,
) -> tuple[int, int]:
    """Return ``(hidden_size, actual_parameters)`` closest to the target budget."""
    return _match(
        lambda hidden: language_model_parameters(
            cell, vocabulary_size, embedding_dim, hidden, num_layers
        ),
        target_parameters,
        minimum,
        maximum,
    )


def match_classifier_hidden_size(
    cell: str,
    target_parameters: int,
    vocabulary_size: int,
    embedding_dim: int,
    num_layers: int = 1,
    num_classes: int = 2,
    minimum: int = 8,
    maximum: int = 4096,
) -> tuple[int, int]:
    """Return ``(hidden_size, actual_parameters)`` closest to the target budget."""
    return _match(
        lambda hidden: sequence_classifier_parameters(
            cell, vocabulary_size, embedding_dim, hidden, num_layers, num_classes
        ),
        target_parameters,
        minimum,
        maximum,
    )
