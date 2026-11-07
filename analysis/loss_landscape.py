"""3-D loss-landscape slices for the trained recurrent checkpoints.

Usage (from the repository root, after a Kaggle or local development run):

    python analysis/loss_landscape.py \
        --task sentiment \
        --run-directory runs/sequence_classification/sentiment/development \
        --output-directory results/recurrent_comparison/landscapes \
        --extra-seeds 1 2 3 4

The script rebuilds the training-time data pipeline from ``config.json`` (see
``trained_models.py``), loads ``checkpoints/<cell>.pt``, and evaluates

    theta(alpha, beta) = theta_0 + alpha * delta_1 + beta * delta_2

on a grid, with filter-normalised directions; the second direction is
orthogonalised against the first per tensor and rescaled to its tensor norm.
The loss is measured on fixed batches of the **evaluation** split the
experiment reported on, so the trained point need not be the minimum of a
slice. It writes one JSON with the raw surfaces and one 3-D PNG with three
panels.

``--extra-seeds`` repeats the slice with other random direction pairs and stores
only their summaries (minimum, centre, maximum, location of the minimum). They
show how much a reading such as "which surface is flattest" depends on the two
directions that happened to be drawn. Random directions depend on the device's
generator, so a surface reproduces only on the same device type (CPU or CUDA).

A surface is a local object: it describes the neighbourhood of one trained
solution along two random directions. It is not evidence of generalisation and
cannot prove an ordering between architectures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from laboratory import (
    count_parameters,
    filter_normalized_direction,
    get_device,
    loss_surface,
    orthogonalize,
)
from trained_models import CELLS, open_comparison


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("sentiment", "generation"), required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=17, help="points per direction")
    parser.add_argument("--batches", type=int, default=4, help="fixed batches per grid point")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0, help="direction seed of the stored surface")
    parser.add_argument(
        "--extra-seeds",
        type=int,
        nargs="*",
        default=[],
        help="further direction seeds whose surfaces are summarised, not stored",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def directions(
    base_state: dict[str, torch.Tensor], seed: int, device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Filter-normalised direction pair; the second is orthogonal to the first."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    first = filter_normalized_direction(base_state, generator)
    return first, orthogonalize(first, filter_normalized_direction(base_state, generator))


def summarise(surface: np.ndarray, grid: np.ndarray) -> dict[str, object]:
    centre = len(grid) // 2
    row, column = np.unravel_index(int(surface.argmin()), surface.shape)
    return {
        "minimum": float(surface.min()),
        "maximum": float(surface.max()),
        "centre": float(surface[centre, centre]),
        "argmin": [float(grid[row]), float(grid[column])],
    }


def plot_surfaces(
    surfaces: dict[str, np.ndarray], grid: np.ndarray, path: Path, task: str, seed: int
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    use_log = max(float(np.nanmax(surface)) for surface in surfaces.values()) > 20.0
    figure = plt.figure(figsize=(17, 5.2))
    for index, (cell, surface) in enumerate(surfaces.items(), start=1):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        a, b = np.meshgrid(grid, grid, indexing="ij")
        z = np.log10(np.maximum(surface, 1e-12)) if use_log else surface
        axis.plot_surface(a, b, z, cmap="viridis", linewidth=0, antialiased=True)
        axis.set_title(
            f"{cell}   min={surface.min():.3f}   centre={surface[len(grid)//2, len(grid)//2]:.3f}"
        )
        axis.set_xlabel("alpha")
        axis.set_ylabel("beta")
        axis.set_zlabel("log10 loss" if use_log else "loss")
        axis.view_init(elev=28, azim=-55)
    figure.suptitle(f"{task}: filter-normalised loss surface (seed {seed})")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    args = parse_arguments()
    device = get_device(args.device)
    comparison = open_comparison(args.task, args.run_directory)
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 128 if args.task == "sentiment" else 16
    batches = comparison.evaluation_batches(args.batches, batch_size, device)

    grid = np.linspace(-1.0, 1.0, args.grid)
    surfaces: dict[str, np.ndarray] = {}
    payload = {
        "task": args.task,
        "grid": grid.tolist(),
        "seed": args.seed,
        "extra_seeds": args.extra_seeds,
        "device": device.type,
        "batches": len(batches),
        "batch_size": batch_size,
        "cells": {},
    }

    for cell in CELLS:
        model = comparison.load(cell, device)
        base_state = {name: value.detach().clone() for name, value in model.state_dict().items()}

        surface = loss_surface(
            model, base_state, *directions(base_state, args.seed, device), grid, batches, comparison.loss
        )
        surfaces[cell] = surface
        payload["cells"][cell] = {
            "parameters": count_parameters(model),
            **summarise(surface, grid),
            "surface": surface.tolist(),
            "extra_seeds": [
                {
                    "seed": seed,
                    **summarise(
                        loss_surface(
                            model,
                            base_state,
                            *directions(base_state, seed, device),
                            grid,
                            batches,
                            comparison.loss,
                        ),
                        grid,
                    ),
                }
                for seed in args.extra_seeds
            ],
        }
        entry = payload["cells"][cell]
        extra = "".join(f" seed{item['seed']}_max={item['maximum']:.4f}" for item in entry["extra_seeds"])
        print(
            f"{cell}: parameters={entry['parameters']:,} min={entry['minimum']:.4f} "
            f"centre={entry['centre']:.4f} max={entry['maximum']:.4f}{extra}"
        )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / f"landscape_{args.task}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    plot_surfaces(
        surfaces,
        grid,
        args.output_directory / f"landscape_{args.task}.png",
        args.task,
        args.seed,
    )
    print(f"wrote {args.output_directory}")


if __name__ == "__main__":
    main()
