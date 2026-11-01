from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from laboratory.training.trainer import Trainer, load_checkpoint, save_checkpoint


def test_trainer_fits_a_linear_regression():
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    trainer = Trainer(model, optimizer, device="cpu")
    inputs = torch.randn(64, 4)
    targets = inputs @ torch.tensor([1.0, 2.0, -1.0, 0.5]) + 0.3
    batch = (inputs, targets)

    def step(current_model, current_batch):
        predictions = current_model(current_batch[0]).squeeze(-1)
        return F.mse_loss(predictions, current_batch[1]), {}

    losses = [trainer.train_step(batch, step)["loss"] for _ in range(80)]
    assert losses[-1] < losses[0] * 0.2


def test_checkpoint_round_trip_restores_weights(tmp_path):
    model = nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    save_checkpoint(tmp_path / "checkpoint.pt", model, optimizer, note="hello")
    expected = model(torch.ones(1, 3)).detach().clone()

    restored = nn.Linear(3, 2)
    metadata = load_checkpoint(tmp_path / "checkpoint.pt", restored)
    assert metadata["note"] == "hello"
    assert torch.allclose(restored(torch.ones(1, 3)), expected, atol=1e-7)
