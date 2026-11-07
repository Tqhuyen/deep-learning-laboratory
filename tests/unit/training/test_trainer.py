from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from laboratory.optimization.muon import Muon
from laboratory.training.trainer import Trainer


def test_trainer_fits_a_linear_regression():
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    optimizer = Muon(model.parameters(), lr=0.15)
    trainer = Trainer(model, optimizer, device="cpu")
    inputs = torch.randn(64, 4)
    targets = inputs @ torch.tensor([1.0, 2.0, -1.0, 0.5]) + 0.3
    batch = (inputs, targets)

    def step(current_model, current_batch):
        predictions = current_model(current_batch[0]).squeeze(-1)
        return F.mse_loss(predictions, current_batch[1]), {}

    losses = [trainer.train_step(batch, step)["loss"] for _ in range(80)]
    assert losses[-1] < losses[0] * 0.2
