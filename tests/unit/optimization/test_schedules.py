from __future__ import annotations

import torch
import torch.nn as nn

from laboratory.optimization.muon import Muon
from laboratory.optimization.schedules import cosine_warmup, noam


def test_cosine_warmup_rises_then_decays_to_the_floor():
    parameter = nn.Parameter(torch.zeros(1))
    optimizer = Muon([parameter], lr=1.0)
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
    optimizer = Muon([parameter], lr=1.0)
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
