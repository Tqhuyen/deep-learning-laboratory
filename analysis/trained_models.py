"""Rebuild the trained models of the recurrent comparison and their evaluation data.

Every analysis script reads a run directory the same way: the tokenizer is
rebuilt exactly as during training, hidden sizes are re-derived from the shared
parameter budget, and ``checkpoints/<cell>.pt`` is loaded per cell. Keeping that
logic here means a script cannot silently evaluate a model on a different
vocabulary or split than the one it was trained on.

Evaluation data is the split the experiment reported on: the SST-2 dev set for
sentiment and the contiguous validation tail of Tiny Shakespeare for generation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
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
    match_classifier_hidden_size,
    match_language_model_hidden_size,
    read_sentiment_tsv,
    stratified_split,
    train_val_split,
)
from laboratory.evaluation.diagnostics import fixed_length_sequences
from laboratory.experiment import data_path, repository_root

CELLS = ("rnn", "gru", "lstm")
SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")


@dataclass
class TrainedComparison:
    """One task of the comparison: how to build, load and evaluate its models."""

    task: str
    config: dict
    run_directory: Path
    build_model: Callable[[str], torch.nn.Module]
    evaluation_batches: Callable[[int, int, torch.device], list]
    loss: Callable[[torch.nn.Module, object], torch.Tensor]
    probe_inputs: Callable[[int, int], torch.Tensor]

    def load(self, cell: str, device: torch.device) -> torch.nn.Module:
        """Model for ``cell`` with its checkpoint weights, in evaluation mode."""
        path = self.run_directory / "checkpoints" / f"{cell}.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing checkpoint {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = self.build_model(cell)
        model.load_state_dict(checkpoint["model"])
        return model.to(device).eval()


def open_comparison(task: str, run_directory: Path) -> TrainedComparison:
    """Read ``config.json`` from a run directory and rebuild the task."""
    run_directory = Path(run_directory)
    config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
    if task == "sentiment":
        return _sentiment(config, run_directory)
    if task == "generation":
        return _generation(config, run_directory)
    raise ValueError(f"unknown task {task!r}; choose 'sentiment' or 'generation'")


def _sentiment(config: dict, run_directory: Path) -> TrainedComparison:
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
        raise RuntimeError(
            f"rebuilt vocabulary has {tokenizer.vocab_size} entries, "
            f"the run used {config['vocabulary_size']}"
        )
    dataset = SentimentDataset(read_sentiment_tsv(dev_path), tokenizer, config["max_len"])

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

    def evaluation_batches(count: int, batch_size: int, device: torch.device) -> list:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: classification_collate(batch, tokenizer.pad_id),
        )
        selected = []
        for index, batch in enumerate(loader):
            if index >= count:
                break
            selected.append({key: value.to(device) for key, value in batch.items()})
        return selected

    def loss(model: torch.nn.Module, batch: dict) -> torch.Tensor:
        return classification_cross_entropy(model(batch["inputs"], batch["lengths"]), batch["labels"])

    def probe_inputs(length: int, count: int) -> torch.Tensor:
        return fixed_length_sequences([ids for ids, _ in dataset.items], length, count)

    return TrainedComparison(
        "sentiment", config, run_directory, build_model, evaluation_batches, loss, probe_inputs
    )


def _generation(config: dict, run_directory: Path) -> TrainedComparison:
    text = data_path("tiny_shakespeare.txt").read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    _, validation_ids = train_val_split(ids, val_fraction=0.1)

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

    def evaluation_batches(count: int, batch_size: int, device: torch.device) -> list:
        dataset = LanguageModelDataset(validation_ids, config["block_size"])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        selected = []
        for index, (inputs, targets) in enumerate(loader):
            if index >= count:
                break
            selected.append((inputs.to(device), targets.to(device)))
        return selected

    def loss(model: torch.nn.Module, batch: tuple) -> torch.Tensor:
        inputs, targets = batch
        return model(inputs, targets)[1]

    def probe_inputs(length: int, count: int) -> torch.Tensor:
        windows = LanguageModelDataset(validation_ids, length)
        return torch.stack([windows[index][0] for index in range(min(count, len(windows)))])

    return TrainedComparison(
        "generation", config, run_directory, build_model, evaluation_batches, loss, probe_inputs
    )
