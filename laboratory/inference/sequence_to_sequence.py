"""Greedy decoding for encoder-decoder models."""

from __future__ import annotations

import torch


@torch.no_grad()
def greedy_decode(
    model,
    src_ids: torch.Tensor,
    max_new_tokens: int,
    src_key_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode from ``<bos>`` until every row emits ``<eos>`` or the budget ends.

    The model must expose ``encode``, ``decode``, ``bos_id``, ``eos_id`` and
    ``max_len`` — the interface of
    :class:`laboratory.models.Seq2SeqTransformer`. Returns the id tensor
    including ``<bos>`` (and ``<eos>`` where it was produced).
    """
    was_training = model.training
    model.eval()
    batch = src_ids.size(0)
    memory = model.encode(src_ids, src_key_padding_mask)
    ys = torch.full((batch, 1), model.bos_id, dtype=torch.long, device=src_ids.device)
    finished = torch.zeros(batch, dtype=torch.bool, device=src_ids.device)

    for _ in range(max_new_tokens):
        logits = model.decode(ys, memory, src_key_padding_mask)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_id], dim=1)
        finished |= next_id.squeeze(1) == model.eos_id
        if bool(finished.all()) or ys.size(1) >= model.max_len:
            break

    model.train(was_training)
    return ys
