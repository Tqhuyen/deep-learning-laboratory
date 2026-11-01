from __future__ import annotations

import pytest

from laboratory.models.capacity import (
    language_model_parameters,
    match_classifier_hidden_size,
    match_language_model_hidden_size,
    sequence_classifier_parameters,
    recurrent_stack_parameters,
)
from laboratory.models.recurrent import RecurrentLanguageModel, RecurrentSequenceClassifier
from laboratory.utilities import count_parameters


@pytest.mark.parametrize("cell", ["rnn", "gru", "lstm"])
def test_language_model_formula_matches_actual_module(cell):
    vocabulary_size, embedding_dim, hidden_size, layers = 17, 8, 12, 2
    model = RecurrentLanguageModel(
        vocabulary_size=vocabulary_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=layers,
        cell=cell,
    )
    expected = language_model_parameters(
        cell, vocabulary_size, embedding_dim, hidden_size, layers
    )
    assert count_parameters(model) == expected


@pytest.mark.parametrize("cell", ["rnn", "gru", "lstm"])
def test_classifier_formula_matches_actual_module(cell):
    vocabulary_size, embedding_dim, hidden_size, layers = 13, 6, 10, 2
    model = RecurrentSequenceClassifier(
        vocabulary_size=vocabulary_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=layers,
        cell=cell,
        pad_id=0,
    )
    expected = sequence_classifier_parameters(
        cell, vocabulary_size, embedding_dim, hidden_size, layers, num_classes=2
    )
    assert count_parameters(model) == expected


def test_gate_multipliers_are_the_paper_counts():
    assert recurrent_stack_parameters("rnn", 10, 20, 1) == (10 * 20 + 20) + (20 * 20 + 20)
    assert recurrent_stack_parameters("gru", 10, 20, 1) == 3 * ((10 * 20 + 20) + (20 * 20 + 20))
    assert recurrent_stack_parameters("lstm", 10, 20, 1) == 4 * ((10 * 20 + 20) + (20 * 20 + 20))


def test_hidden_size_matching_lands_near_the_target():
    hidden, actual = match_language_model_hidden_size(
        "lstm", target_parameters=200_000, vocabulary_size=65, embedding_dim=64
    )
    assert actual == language_model_parameters("lstm", 65, 64, hidden)
    assert abs(actual - 200_000) / 200_000 < 0.05

    hidden, actual = match_classifier_hidden_size(
        "gru", target_parameters=500_000, vocabulary_size=2000, embedding_dim=64
    )
    assert actual == sequence_classifier_parameters("gru", 2000, 64, hidden)
    assert abs(actual - 500_000) / 500_000 < 0.05
