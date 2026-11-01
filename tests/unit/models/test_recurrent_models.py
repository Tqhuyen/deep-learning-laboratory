from __future__ import annotations

import torch

from laboratory.models.recurrent import (
    RecurrentLanguageModel,
    RecurrentSequenceClassifier,
)


def test_classifier_ignores_padding_content_after_the_real_sequence():
    model = RecurrentSequenceClassifier(
        vocabulary_size=20,
        embedding_dim=8,
        hidden_size=10,
        cell="lstm",
        pad_id=0,
    )
    model.eval()
    inputs = torch.randint(1, 20, (2, 7))
    lengths = torch.tensor([7, 4])
    modified = inputs.clone()
    modified[1, 4:] = torch.randint(1, 20, (3,))

    with torch.no_grad():
        original = model(inputs, lengths)
        changed = model(modified, lengths)
    assert torch.allclose(original[1], changed[1], atol=1e-6)


def test_language_model_returns_logits_and_loss():
    model = RecurrentLanguageModel(
        vocabulary_size=11, embedding_dim=8, hidden_size=10, cell="gru"
    )
    inputs = torch.randint(0, 11, (2, 6))
    logits, loss = model(inputs, inputs)
    assert logits.shape == (2, 6, 11)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert model.embedding.weight.grad is not None
