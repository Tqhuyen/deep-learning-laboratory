from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from laboratory.training import (
    RunningMean,
    Trainer,
    accuracy,
    cosine_warmup,
    load_checkpoint,
    noam,
    save_checkpoint,
)


def test_cosine_warmup_rises_then_decays_to_the_floor():
    parameter = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = cosine_warmup(optimizer, warmup_steps=10, total_steps=110, min_lr_ratio=0.01)

    learning_rates = []
    for _ in range(110):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])

    assert learning_rates[0] == 0.2  # (step + 1) / warmup_steps
    assert max(learning_rates) <= 1.0 + 1e-6
    assert abs(learning_rates[-1] - 0.01) < 1e-6
    assert learning_rates[50] > learning_rates[-1]


def test_noam_peaks_at_the_warmup_boundary():
    parameter = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = noam(optimizer, d_model=512, warmup_steps=40)

    learning_rates = []
    for _ in range(80):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])

    peak = max(range(len(learning_rates)), key=learning_rates.__getitem__)
    assert abs(learning_rates[peak] - (512 * 40) ** -0.5) < 1e-7
    assert peak == 39
    assert learning_rates[-1] < learning_rates[peak]


def test_accuracy_ignores_padding_positions():
    logits = torch.tensor([[[0.0, 1.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]]])
    targets = torch.tensor([[1, 0, 0, -100]])
    assert accuracy(logits, targets, ignore_index=-100) == 1.0


def test_running_mean_is_weighted_by_batch_size():
    mean = RunningMean()
    mean.update(2.0, n=3)
    mean.update(4.0, n=1)
    assert mean.mean == 2.5


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
