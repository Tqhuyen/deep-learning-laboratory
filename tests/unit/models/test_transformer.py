from __future__ import annotations

import torch

from laboratory.models.transformer import GPT, Seq2SeqTransformer, TransformerEncoder


def test_gpt_returns_logits_and_a_finite_loss():
    model = GPT(vocab_size=11, d_model=16, n_heads=2, n_layers=1, max_len=8)
    inputs = torch.randint(0, 11, (2, 8))
    logits, loss = model(inputs, inputs)
    assert logits.shape == (2, 8, 11)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert model.token_embedding.weight.grad is not None


def test_gpt_is_causal():
    model = GPT(vocab_size=13, d_model=16, n_heads=2, n_layers=2, max_len=10, dropout=0.0)
    model.eval()
    inputs = torch.randint(0, 13, (1, 10))
    modified = inputs.clone()
    modified[0, -1] = (modified[0, -1] + 1) % 13
    with torch.no_grad():
        original_logits, _ = model(inputs)
        modified_logits, _ = model(modified)
    assert torch.allclose(original_logits[:, :-1], modified_logits[:, :-1], atol=1e-6)


def test_encoder_hides_padding_from_real_tokens():
    model = TransformerEncoder(
        vocab_size=17, d_model=16, n_heads=2, n_layers=2, max_len=8, dropout=0.0
    )
    model.eval()
    inputs = torch.randint(0, 17, (1, 5))
    modified = inputs.clone()
    modified[0, -1] = (modified[0, -1] + 1) % 17
    padding = torch.tensor([[True, True, True, True, False]])
    with torch.no_grad():
        original = model(inputs, key_padding_mask=padding)
        changed = model(modified, key_padding_mask=padding)
    assert torch.allclose(original[:, :4], changed[:, :4], atol=1e-6)


def test_sequence_to_sequence_forward_shape():
    model = Seq2SeqTransformer(
        src_vocab_size=14,
        tgt_vocab_size=14,
        d_model=16,
        n_heads=2,
        n_layers=1,
        max_len=12,
        dropout=0.0,
        bos_id=1,
        eos_id=2,
        pad_id=0,
    )
    source = torch.randint(3, 14, (2, 5))
    target = torch.randint(3, 14, (2, 4))
    source_padding = torch.ones(2, 5, dtype=torch.bool)
    source_padding[1, 4] = False

    logits = model(source, target, src_key_padding_mask=source_padding)
    assert logits.shape == (2, 4, 14)
