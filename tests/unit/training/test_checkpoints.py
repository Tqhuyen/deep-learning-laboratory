from __future__ import annotations

import torch
import torch.nn as nn

from laboratory.training.checkpoints import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip_restores_weights(tmp_path):
    model = nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    save_checkpoint(tmp_path / "checkpoint.pt", model, optimizer, note="hello")
    expected = model(torch.ones(1, 3)).detach().clone()

    restored = nn.Linear(3, 2)
    metadata = load_checkpoint(tmp_path / "checkpoint.pt", restored)
    assert metadata["note"] == "hello"
    assert torch.allclose(restored(torch.ones(1, 3)), expected, atol=1e-7)
