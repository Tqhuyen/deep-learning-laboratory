"""Gradient reach and gate statistics of the trained recurrent checkpoints.

Usage (from the repository root):

    python analysis/gradient_reach.py --task generation \
        --run-directory runs/language_modeling/recurrent_generation/development \
        --output-directory results/recurrent_comparison/reach

For every cell the script measures, on unpadded evaluation sequences of
``--length`` tokens,

    g(k) = mean over sequences of || d ||h_T||^2 / d x_(T-k) ||,

fits the slope of log g(k) against the lag k, and records the quantities that
the theory says control it:

* RNN: ``||dh_k/dh_(k-1)|| = ||diag(1 - h_k^2) W_hh|| <= sigma_max(W_hh)`` per
  step, so the spectral norm of the recurrent matrix and the mean tanh
  derivative are reported;
* GRU: ``h_k = (1 - z_k) * n_k + z_k * h_(k-1)``, so the update gate ``z`` is
  the per-step factor on the direct carry path;
* LSTM: ``c_k = f_k * c_(k-1) + i_k * g_k``, so the forget gate ``f`` is the
  per-step factor on the cell path.

The gate statistics describe one path through the network; the measured reach
sums all paths. They are reported side by side so a mechanistic explanation can
be checked against the measurement, not to replace it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from laboratory import count_parameters, get_device
from laboratory.evaluation.diagnostics import decay_slope, retention, timestep_gradient_norms
from trained_models import CELLS, open_comparison

DEFAULT_LENGTH = {"sentiment": 24, "generation": 128}
DEFAULT_COUNT = {"sentiment": 128, "generation": 64}
LAGS = (1, 2, 4, 8, 16, 32, 64, 127)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("sentiment", "generation"), required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--length", type=int, default=None, help="probe sequence length")
    parser.add_argument("--count", type=int, default=None, help="probe sequences")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


@torch.no_grad()
def per_step_factors(cell: str, recurrent: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Unroll the first layer and return the per-step factor of the carry path.

    Shape ``(batch, time, hidden)``: ``1 - h^2`` for the RNN (tanh derivative),
    the update gate ``z`` for the GRU and the forget gate ``f`` for the LSTM.
    """
    layer = recurrent.cells[0]
    batch, steps, _ = inputs.shape
    hidden = layer.linear_hh.in_features
    h = inputs.new_zeros(batch, hidden)
    c = inputs.new_zeros(batch, hidden)
    factors = []
    for step in range(steps):
        x = inputs[:, step]
        if cell == "rnn":
            h = layer(x, h)
            factors.append(1.0 - h.pow(2))
        elif cell == "gru":
            update = torch.sigmoid(
                layer.linear_ih(x).chunk(3, dim=-1)[1] + layer.linear_hh(h).chunk(3, dim=-1)[1]
            )
            factors.append(update)
            h = layer(x, h)
        else:
            gates = (layer.linear_ih(x) + layer.linear_hh(h)).chunk(4, dim=-1)
            factors.append(torch.sigmoid(gates[1]))
            h, (_, c) = layer(x, (h, c))
    return torch.stack(factors, dim=1)


def summarise(values: torch.Tensor) -> dict[str, float]:
    flat = values.double().flatten()
    return {
        "mean": float(flat.mean()),
        "geometric_mean": float(flat.clamp_min(1e-12).log().mean().exp()),
        "fraction_above_0.9": float((flat > 0.9).double().mean()),
        "fraction_above_0.99": float((flat > 0.99).double().mean()),
        "unit_mean_percentile_90": float(
            values.double().mean(dim=(0, 1)).quantile(0.9)
        ),
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    args = parse_arguments()
    device = get_device(args.device)
    comparison = open_comparison(args.task, args.run_directory)
    length = args.length or DEFAULT_LENGTH[args.task]
    count = args.count or DEFAULT_COUNT[args.task]
    tokens = comparison.probe_inputs(length, count).to(device)

    payload = {
        "task": args.task,
        "probe": {
            "length": int(tokens.shape[1]),
            "sequences": int(tokens.shape[0]),
            "quantity": "mean over sequences of || d ||h_T||^2 / d x_t ||",
        },
        "cells": {},
    }
    for cell in CELLS:
        model = comparison.load(cell, device)
        embedded = model.embedding(tokens).detach()
        norms = timestep_gradient_norms(model.recurrent, embedded).double().numpy()
        slope = decay_slope(norms)
        entry = {
            "hidden_size": int(model.recurrent.cells[0].linear_hh.in_features),
            "parameters": count_parameters(model),
            "norms": norms.tolist(),
            "slope": slope,
            "per_step_retention": math.exp(slope) if math.isfinite(slope) else None,
            "half_life_steps": math.log(2.0) / -slope if math.isfinite(slope) and slope < 0 else None,
            "retention": retention(norms, [lag for lag in LAGS if lag < len(norms)]),
            "carry_factor": summarise(per_step_factors(cell, model.recurrent, embedded)),
        }
        if cell == "rnn":
            entry["recurrent_spectral_norm"] = float(
                torch.linalg.matrix_norm(model.recurrent.cells[0].linear_hh.weight.detach().double(), ord=2)
            )
        payload["cells"][cell] = entry
        carry = entry["carry_factor"]
        extra = (
            f" sigma_max(W_hh)={entry['recurrent_spectral_norm']:.3f}" if cell == "rnn" else ""
        )
        print(
            f"{cell}: slope={slope:.4f} retention/step={entry['per_step_retention']:.4f} "
            f"half-life={entry['half_life_steps']} steps "
            f"carry mean={carry['mean']:.3f} geo={carry['geometric_mean']:.3f} "
            f">0.9={carry['fraction_above_0.9']:.3f}{extra}"
        )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / f"reach_{args.task}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    for cell, entry in payload["cells"].items():
        values = np.asarray(entry["norms"], dtype=float)[::-1]
        values = np.where(values > values.max() * 1e-12, values, np.nan)
        axis.semilogy(np.arange(values.size), values, linewidth=1.6, label=f"{cell} (slope {entry['slope']:.3f})")
    axis.set_xlabel("lag k from the final timestep")
    axis.set_ylabel("mean ||d ||h_T||^2 / d x_(T-k)||  (log scale)")
    axis.set_title(
        f"{args.task}: gradient reach on {tokens.shape[0]} unpadded sequences of {tokens.shape[1]} tokens"
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(args.output_directory / f"reach_{args.task}.png", dpi=160)
    plt.close(figure)
    print(f"wrote {args.output_directory}")


if __name__ == "__main__":
    main()
