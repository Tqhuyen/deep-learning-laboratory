"""Decoding: turn model outputs into next tokens."""

from __future__ import annotations

import torch


def next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Pick one token id per row from last-position ``(batch, vocabulary)`` logits.

    ``temperature=0`` is greedy; otherwise top-k filtering is applied before
    sampling. Returned shape is ``(batch, 1)``.
    """
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / max(temperature, 1e-8)
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        cutoff = torch.topk(logits, k, dim=-1).values[:, [-1]]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))
    probabilities = torch.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


@torch.no_grad()
def generate(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    context_window: int | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Autoregressively extend ``prompt_ids`` (batch, tokens).

    The model must accept an id tensor and return logits or ``(logits, ...)``;
    that covers both :class:`laboratory.models.GPT` and a recurrent language
    model. ``context_window`` limits the visible history (default: the model's
    ``max_len`` when it has one).
    """
    if context_window is None:
        context_window = getattr(model, "max_len", None)

    was_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        window = prompt_ids if context_window is None else prompt_ids[:, -context_window:]
        outputs = model(window)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        prompt_ids = torch.cat(
            [prompt_ids, next_token(logits[:, -1, :], temperature, top_k)], dim=1
        )
    model.train(was_training)
    return prompt_ids
