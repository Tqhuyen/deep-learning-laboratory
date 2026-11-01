from __future__ import annotations

import torch

from laboratory.inference.generation import generate, next_token
from laboratory.models.transformer import GPT


def test_generate_appends_tokens_and_preserves_the_prompt():
    model = GPT(vocab_size=9, d_model=8, n_heads=2, n_layers=1, max_len=6)
    prompt = torch.randint(0, 9, (2, 3))
    generated = generate(model, prompt, max_new_tokens=4, temperature=0.0)
    assert generated.shape == (2, 7)
    assert torch.equal(generated[:, :3], prompt)
    assert torch.equal(
        generated, generate(model, prompt, max_new_tokens=4, temperature=0.0)
    )


def test_generate_crops_history_longer_than_the_model_window():
    model = GPT(vocab_size=9, d_model=8, n_heads=2, n_layers=1, max_len=6)
    prompt = torch.randint(0, 9, (1, 10))
    generated = generate(model, prompt, max_new_tokens=2, temperature=0.0)
    assert generated.shape == (1, 12)


def test_next_token_with_zero_temperature_is_greedy():
    logits = torch.tensor([[0.1, 2.0, -1.0]])
    assert next_token(logits, temperature=0).item() == 1


def test_next_token_top_k_never_selects_outside_the_cutoff():
    logits = torch.tensor([[10.0, 9.0, -100.0, -100.0]])
    sampled = {next_token(logits, temperature=1.0, top_k=2).item() for _ in range(50)}
    assert sampled <= {0, 1}
