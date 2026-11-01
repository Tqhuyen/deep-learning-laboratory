"""3-D loss-landscape slices for the trained recurrent checkpoints.

Usage (from the repository root, after a Kaggle or local development run):

    python analysis/loss_landscape.py \
        --task sentiment \
        --run-directory runs/sequence_classification/sentiment/development \
        --output-directory results/recurrent_comparison/landscapes

The script rebuilds the training-time data pipeline from ``config.json``,
loads ``checkpoints/<cell>.pt``, and evaluates

    theta(alpha, beta) = theta_0 + alpha * delta_1 + beta * delta_2

on a grid, with filter-normalised orthogonal directions. It writes one JSON
with the raw surfaces and one 3-D PNG with three panels.

A surface is a local object: it describes the neighbourhood of one trained
solution. It is not evidence of generalisation and cannot prove an ordering
between architectures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from laboratory import (
    CharTokenizer,
    LanguageModelDataset,
    RecurrentLanguageModel,
    RecurrentSequenceClassifier,
    SentimentDataset,
    WordTokenizer,
    classification_collate,
    classification_cross_entropy,
    count_parameters,
    filter_normalized_direction,
    get_device,
    loss_surface,
    match_classifier_hidden_size,
    match_language_model_hidden_size,
    orthogonalize,
    read_sentiment_tsv,
    stratified_split,
    train_val_split,
)
from laboratory.experiment import data_path, repository_root

CELLS = ("rnn", "gru", "lstm")
SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("sentiment", "generation"), required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=17, help="points per direction")
    parser.add_argument("--batches", type=int, default=4, help="fixed batches per grid point")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_config(run_directory: Path) -> dict:
    return json.loads((run_directory / "config.json").read_text(encoding="utf-8"))


def sentiment_pipeline(config: dict, device: torch.device, batches: int, batch_size: int):
    root = repository_root()
    train_path, dev_path = root / "data/sst2/train.tsv", root / "data/sst2/dev.tsv"
    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError("run `python data/prepare_sst2.py` first")

    pool = read_sentiment_tsv(train_path)
    if config["train_limit"] is not None:
        rng = np.random.default_rng(0)
        indices = rng.permutation(len(pool))[: config["train_limit"]]
        pool = [pool[index] for index in sorted(indices)]
    train_pool, _ = stratified_split(pool, test_fraction=config["val_fraction"], seed=0)
    tokenizer = WordTokenizer.from_text(
        " ".join(text for text, _ in train_pool),
        special_tokens=SPECIAL_TOKENS,
        min_freq=config["min_frequency"],
    )
    if tokenizer.vocab_size != config["vocabulary_size"]:
        raise RuntimeError("rebuilt vocabulary does not match the checkpoint config")

    dataset = SentimentDataset(read_sentiment_tsv(dev_path), tokenizer, config["max_len"])
    collate = lambda batch: classification_collate(batch, tokenizer.pad_id)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    selected = []
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        selected.append({key: value.to(device) for key, value in batch.items()})

    def build_model(cell: str) -> RecurrentSequenceClassifier:
        hidden, _ = match_classifier_hidden_size(
            cell,
            target_parameters=config["target_parameters"],
            vocabulary_size=tokenizer.vocab_size,
            embedding_dim=config["embedding_dim"],
        )
        return RecurrentSequenceClassifier(
            vocabulary_size=tokenizer.vocab_size,
            embedding_dim=config["embedding_dim"],
            hidden_size=hidden,
            cell=cell,
            pad_id=tokenizer.pad_id,
        )

    def step(model, batch):
        logits = model(batch["inputs"], batch["lengths"])
        return classification_cross_entropy(logits, batch["labels"])

    return build_model, selected, step


def generation_pipeline(config: dict, device: torch.device, batches: int, batch_size: int):
    text = data_path("tiny_shakespeare.txt").read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    _, validation_ids = train_val_split(ids, val_fraction=0.1)
    dataset = LanguageModelDataset(validation_ids, config["block_size"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    selected = []
    for index, (inputs, targets) in enumerate(loader):
        if index >= batches:
            break
        selected.append((inputs.to(device), targets.to(device)))

    def build_model(cell: str) -> RecurrentLanguageModel:
        hidden, _ = match_language_model_hidden_size(
            cell,
            target_parameters=config["target_parameters"],
            vocabulary_size=tokenizer.vocab_size,
            embedding_dim=config["embedding_dim"],
        )
        return RecurrentLanguageModel(
            vocabulary_size=tokenizer.vocab_size,
            embedding_dim=config["embedding_dim"],
            hidden_size=hidden,
            cell=cell,
        )

    def step(model, batch):
        inputs, targets = batch
        _, loss = model(inputs, targets)
        return loss

    return build_model, selected, step


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
    config = load_config(args.run_directory)
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 128 if args.task == "sentiment" else 16
    build_model, batches, step = (
        sentiment_pipeline(config, device, args.batches, batch_size)
        if args.task == "sentiment"
        else generation_pipeline(config, device, args.batches, batch_size)
    )

    grid = np.linspace(-1.0, 1.0, args.grid)
    surfaces: dict[str, np.ndarray] = {}
    payload = {"task": args.task, "grid": grid.tolist(), "cells": {}}

    for cell in CELLS:
        checkpoint_path = args.run_directory / "checkpoints" / f"{cell}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"missing {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = build_model(cell).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        base_state = {name: value.detach().clone() for name, value in model.state_dict().items()}

        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)
        first = filter_normalized_direction(base_state, generator)
        second = filter_normalized_direction(base_state, generator)
        second = orthogonalize(first, second)

        surface = loss_surface(model, base_state, first, second, grid, batches, step)
        surfaces[cell] = surface
        payload["cells"][cell] = {
            "parameters": count_parameters(model),
            "minimum": float(surface.min()),
            "maximum": float(surface.max()),
            "centre": float(surface[len(grid) // 2, len(grid) // 2]),
            "surface": surface.tolist(),
        }
        print(
            f"{cell}: parameters={count_parameters(model):,} "
            f"min={surface.min():.4f} centre={surface[len(grid)//2, len(grid)//2]:.4f} "
            f"max={surface.max():.4f}"
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
