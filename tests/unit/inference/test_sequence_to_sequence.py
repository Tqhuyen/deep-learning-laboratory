from __future__ import annotations

import torch

from laboratory.inference.sequence_to_sequence import greedy_decode
from laboratory.models.transformer import Seq2SeqTransformer


def make_model(max_len: int) -> Seq2SeqTransformer:
    return Seq2SeqTransformer(
        src_vocab_size=14,
        tgt_vocab_size=14,
        d_model=16,
        n_heads=2,
        n_layers=1,
        max_len=max_len,
        dropout=0.0,
        bos_id=1,
        eos_id=2,
        pad_id=0,
    )


def test_greedy_decode_starts_with_bos_and_respects_the_budget():
    model = make_model(max_len=12)
    source = torch.randint(3, 14, (2, 5))
    source_padding = torch.ones(2, 5, dtype=torch.bool)
    source_padding[1, 4] = False

    generated = greedy_decode(
        model, source, max_new_tokens=6, src_key_padding_mask=source_padding
    )
    assert generated.shape[0] == 2
    assert torch.all(generated[:, 0] == 1)
    assert generated.size(1) <= 7
    assert torch.equal(
        generated,
        greedy_decode(model, source, max_new_tokens=6, src_key_padding_mask=source_padding),
    )


def test_greedy_decode_stops_at_the_model_sequence_limit():
    model = make_model(max_len=4)
    source = torch.randint(3, 14, (1, 3))
    generated = greedy_decode(model, source, max_new_tokens=10)
    assert generated.size(1) <= 4
