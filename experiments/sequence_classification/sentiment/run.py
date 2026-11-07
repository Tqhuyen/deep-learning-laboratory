"""Sentiment classification: RNN vs GRU vs LSTM at matched parameter budgets.

Question: at the same number of parameters, the same training budget and the
same data, which recurrent cell classifies sentiment best, and can the
difference be explained by optimisation mechanics rather than noise?

Design:

* SST-2 (GLUE). ``tiny``/``smoke`` use a committed 160-example fixture so the
  CPU smoke test is offline; ``development``/``full`` use the prepared SST-2
  splits (``python data/prepare_sst2.py``).
* Each cell is sized by :func:`laboratory.match_classifier_hidden_size` to hit
  the same parameter target. Per-layer cell parameters are
  ``m * (I*H + H^2 + 2H)`` with ``m = 1`` (RNN), ``3`` (GRU), ``4`` (LSTM).
* Identical optimiser, schedule, batch size and step budget for every cell,
  three seeds for ``development`` and above.
* Statistics: mean ± std across seeds, paired bootstrap over test examples
  (seed-averaged correctness) and exact McNemar on the seed-0 predictions,
  Holm-corrected across the three pairs.
* Mechanism: ``|| d ||h_T||^2 / d x_t ||`` for every input position ``t`` of
  unpadded test sentences truncated to ``probe_length`` tokens, i.e. how
  strongly the final state still depends on each earlier input.

Run:

    python data/prepare_sst2.py                     # only for development/full
    python experiments/sequence_classification/sentiment/run.py --tier tiny
    python experiments/sequence_classification/sentiment/run.py --tier development
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from laboratory import (
    RecurrentSequenceClassifier,
    SentimentDataset,
    Trainer,
    WordTokenizer,
    accuracy,
    classification_collate,
    classification_cross_entropy,
    cosine_warmup,
    count_parameters,
    decay_slope,
    fixed_length_sequences,
    get_device,
    holm_adjust,
    make_optimizer,
    match_classifier_hidden_size,
    mcnemar_exact,
    paired_bootstrap_difference,
    read_sentiment_tsv,
    save_checkpoint,
    set_seed,
    timestep_gradient_norms,
    stratified_split,
)
from laboratory.evaluation.plots import plot_bars, plot_gradient_decay, plot_lines
from laboratory.experiment import (
    argument_parser,
    data_path,
    environment_summary,
    prepare_run,
    repository_root,
    select_preset,
    write_config,
    write_metrics,
)

AREA = "sequence_classification"
NAME = "sentiment"

CELLS = ("rnn", "gru", "lstm")
SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")

PRESETS = {
    "tiny": dict(
        source="fixture",
        max_steps=6,
        batch_size=8,
        max_len=24,
        embedding_dim=32,
        target_parameters=150_000,
        min_frequency=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        seeds=(0,),
        eval_every=2,
        validation_batches=1,
        val_fraction=0.2,
        length_buckets=(0, 8, 16, 24),
        bootstrap_repetitions=400,
        probe_length=8,
        probe_examples=16,
    ),
    "smoke": dict(
        source="fixture",
        max_steps=120,
        batch_size=16,
        max_len=32,
        embedding_dim=64,
        target_parameters=300_000,
        min_frequency=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        seeds=(0,),
        eval_every=20,
        validation_batches=2,
        val_fraction=0.2,
        length_buckets=(0, 8, 16, 32),
        bootstrap_repetitions=1000,
        probe_length=12,
        probe_examples=32,
    ),
    "development": dict(
        source="sst2",
        train_limit=20_000,
        max_steps=1500,
        batch_size=64,
        max_len=64,
        embedding_dim=64,
        target_parameters=2_000_000,
        min_frequency=2,
        learning_rate=3e-4,
        weight_decay=0.01,
        seeds=(0, 1, 2),
        eval_every=100,
        validation_batches=10,
        val_fraction=0.1,
        length_buckets=(0, 8, 16, 32, 64),
        bootstrap_repetitions=2000,
        probe_length=24,
        probe_examples=128,
    ),
    "full": dict(
        source="sst2",
        train_limit=None,
        max_steps=5000,
        batch_size=128,
        max_len=64,
        embedding_dim=128,
        target_parameters=4_000_000,
        min_frequency=2,
        learning_rate=3e-4,
        weight_decay=0.01,
        seeds=(0, 1, 2, 3, 4),
        eval_every=250,
        validation_batches=20,
        val_fraction=0.05,
        length_buckets=(0, 8, 16, 32, 64),
        bootstrap_repetitions=3000,
        probe_length=24,
        probe_examples=256,
    ),
}


def load_examples(config: dict) -> tuple[list, list, list]:
    """Return ``(train, validation, test)`` examples for the configured source."""
    fixture = read_sentiment_tsv(data_path("sentiment_fixture.tsv"))
    if config["source"] == "fixture":
        train_validation, test = stratified_split(fixture, test_fraction=0.25, seed=0)
        train, validation = stratified_split(
            train_validation, test_fraction=config["val_fraction"], seed=1
        )
        return train, validation, test

    sst2_dir = repository_root() / "data" / "sst2"
    train_path, dev_path = sst2_dir / "train.tsv", sst2_dir / "dev.tsv"
    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError(
            f"{sst2_dir} is empty. Run `python data/prepare_sst2.py` first."
        )
    pool = read_sentiment_tsv(train_path)
    if config["train_limit"] is not None:
        rng = np.random.default_rng(0)
        indices = rng.permutation(len(pool))[: config["train_limit"]]
        pool = [pool[index] for index in sorted(indices)]
    train, validation = stratified_split(
        pool, test_fraction=config["val_fraction"], seed=0
    )
    return train, validation, read_sentiment_tsv(dev_path)


def build_loaders(
    train_examples, validation_examples, test_examples, tokenizer, config
):
    def collate(batch):
        return classification_collate(batch, tokenizer.pad_id)

    train = SentimentDataset(train_examples, tokenizer, config["max_len"])
    validation = SentimentDataset(validation_examples, tokenizer, config["max_len"])
    test = SentimentDataset(test_examples, tokenizer, config["max_len"])

    train_loader = DataLoader(
        train,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        validation,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate,
    )
    return train_loader, validation_loader, test_loader, test


def make_step():
    def step(model, batch):
        logits = model(batch["inputs"], batch["lengths"])
        loss = classification_cross_entropy(logits, batch["labels"])
        return loss, {"accuracy": accuracy(logits.detach(), batch["labels"])}

    return step


@torch.no_grad()
def predict(model, loader, device):
    """Return per-example correctness, NLL and predictions on a loader."""
    model.eval()
    correctness, negative_log_likelihood, predictions, labels = [], [], [], []
    for batch in loader:
        inputs = batch["inputs"].to(device)
        lengths = batch["lengths"].to(device)
        targets = batch["labels"].to(device)
        logits = model(inputs, lengths)
        per_example = F.cross_entropy(logits, targets, reduction="none")
        correctness.append((logits.argmax(-1) == targets).cpu().numpy())
        negative_log_likelihood.append(per_example.cpu().numpy())
        predictions.append(logits.argmax(-1).cpu().numpy())
        labels.append(targets.cpu().numpy())
    return (
        np.concatenate(correctness),
        np.concatenate(negative_log_likelihood),
        np.concatenate(predictions),
        np.concatenate(labels),
    )


def train_one(cell: str, seed: int, tokenizer, loaders, config, device, checkpoint_path=None):
    train_loader, validation_loader, test_loader, test_dataset = loaders
    set_seed(seed)

    hidden_size, expected_parameters = match_classifier_hidden_size(
        cell,
        target_parameters=config["target_parameters"],
        vocabulary_size=tokenizer.vocab_size,
        embedding_dim=config["embedding_dim"],
    )
    model = RecurrentSequenceClassifier(
        vocabulary_size=tokenizer.vocab_size,
        embedding_dim=config["embedding_dim"],
        hidden_size=hidden_size,
        num_layers=1,
        cell=cell,
        num_classes=2,
        pad_id=tokenizer.pad_id,
    )
    actual_parameters = count_parameters(model)
    if abs(actual_parameters - expected_parameters) > 0:
        raise RuntimeError(
            f"capacity formula mismatch for {cell}: {actual_parameters} vs {expected_parameters}"
        )

    optimizer = make_optimizer(
        model.parameters(),
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = cosine_warmup(
        optimizer,
        warmup_steps=max(1, config["max_steps"] // 20),
        total_steps=config["max_steps"],
    )
    trainer = Trainer(model, optimizer, device=device, scheduler=scheduler)
    step_fn = make_step()

    history = []
    iterator = itertools.cycle(train_loader)
    for step in range(1, config["max_steps"] + 1):
        logs = trainer.train_step(next(iterator), step_fn)
        if step % config["eval_every"] == 0 or step == config["max_steps"]:
            validation = trainer.evaluate(
                validation_loader, step_fn, max_batches=config["validation_batches"]
            )
            history.append(
                {
                    "step": step,
                    "train_loss": logs["loss"],
                    "validation_loss": validation["loss"],
                    "validation_accuracy": validation["accuracy"],
                }
            )
            print(
                f"{cell} seed={seed} step={step} "
                f"train_loss={logs['loss']:.4f} "
                f"val_loss={validation['loss']:.4f} "
                f"val_acc={validation['accuracy']:.4f}"
            )

    correctness, nll, predictions, labels = predict(model, test_loader, device)
    test_accuracy = float(correctness.mean())
    print(f"{cell} seed={seed} test_accuracy={test_accuracy:.4f}")

    # Mechanism probe on unpadded sentences: the final timestep must be a real
    # token, otherwise "h_T" is the state after the padding.
    probe_inputs = fixed_length_sequences(
        [ids for ids, _ in test_dataset.items],
        config["probe_length"],
        config["probe_examples"],
    ).to(device)
    model.eval()
    embeddings = model.embedding(probe_inputs).detach()
    gradient_norms = timestep_gradient_norms(model.recurrent, embeddings).numpy()

    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path, model, cell=cell, seed=seed)

    return {
        "cell": cell,
        "seed": seed,
        "hidden_size": hidden_size,
        "parameters": actual_parameters,
        "history": history,
        "accuracy": test_accuracy,
        "nll": float(nll.mean()),
        "correctness": correctness.tolist(),
        "predictions": predictions.tolist(),
        "labels": labels.tolist(),
        "gradient_norms": gradient_norms.tolist(),
        "gradient_slope": decay_slope(gradient_norms),
        "token_lengths": [len(test_dataset[index][0]) for index in range(len(test_dataset))],
    }


def aggregate(results: dict[str, list[dict]], config: dict) -> dict:
    cells = list(results)
    summary = {
        "parameters": {},
        "metrics": {},
        "pairwise": [],
        "gradient_slopes": {},
    }
    for cell in cells:
        runs = results[cell]
        summary["parameters"][cell] = {
            "hidden_size": runs[0]["hidden_size"],
            "parameters": runs[0]["parameters"],
        }
        accuracies = [run["accuracy"] for run in runs]
        nlls = [run["nll"] for run in runs]
        summary["metrics"][cell] = {
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0,
            "nll_mean": float(np.mean(nlls)),
            "nll_std": float(np.std(nlls, ddof=1)) if len(nlls) > 1 else 0.0,
            "values": [float(value) for value in accuracies],
        }
        summary["gradient_slopes"][cell] = float(np.mean([run["gradient_slope"] for run in runs]))

    pairs = [("rnn", "gru"), ("rnn", "lstm"), ("gru", "lstm")]
    raw_p_values = []
    for first, second in pairs:
        first_runs, second_runs = results[first], results[second]
        first_correct = np.mean([run["correctness"] for run in first_runs], axis=0)
        second_correct = np.mean([run["correctness"] for run in second_runs], axis=0)
        interval = paired_bootstrap_difference(
            first_correct,
            second_correct,
            repetitions=config["bootstrap_repetitions"],
            seed=0,
        )
        p_value = mcnemar_exact(
            np.asarray(first_runs[0]["correctness"], dtype=bool),
            np.asarray(second_runs[0]["correctness"], dtype=bool),
        )
        raw_p_values.append(p_value)
        summary["pairwise"].append(
            {
                "first": first,
                "second": second,
                "accuracy_difference": float(first_correct.mean() - second_correct.mean()),
                "bootstrap_interval": list(interval),
                "mcnemar_p": p_value,
            }
        )
    adjusted = holm_adjust(raw_p_values)
    for entry, corrected in zip(summary["pairwise"], adjusted):
        entry["holm_p"] = corrected
    summary["pairwise_bootstrap_repetitions"] = config["bootstrap_repetitions"]
    return summary


def accuracy_by_length(results: dict[str, list[dict]], config: dict):
    cells = list(results)
    lengths = np.asarray(results[cells[0]][0]["token_lengths"])
    edges = list(config["length_buckets"])
    bucket = np.digitize(lengths, edges[1:-1])
    series = {}
    for cell in cells:
        correctness = np.mean(
            [run["correctness"] for run in results[cell]], axis=0
        )
        values = [
            float(correctness[bucket == index].mean()) if np.any(bucket == index) else float("nan")
            for index in range(len(edges) - 1)
        ]
        series[cell] = values
    counts = [int((bucket == index).sum()) for index in range(len(edges) - 1)]
    return series, counts, edges


def write_report(
    run_directory: Path,
    summary: dict,
    config: dict,
    tier: str,
    notes: tuple[str, ...] = (),
) -> None:
    lines = [
        "# Sentiment classification: RNN vs GRU vs LSTM",
        "",
        f"Tier: `{tier}`. Seeds: {config['seeds']}.",
        "",
        "Parameter matching: per-layer cell parameters are",
        "`m * (I*H + H^2 + 2H)` with `m = 1` (RNN), `3` (GRU), `4` (LSTM);",
        "the hidden size is solved for the shared budget.",
        "",
        "| cell | hidden | parameters | accuracy (mean ± std) | NLL | gradient slope |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cell, parameters in summary["parameters"].items():
        metrics = summary["metrics"][cell]
        lines.append(
            f"| {cell} | {parameters['hidden_size']} | {parameters['parameters']:,} | "
            f"{metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f} | "
            f"{metrics['nll_mean']:.4f} | {summary['gradient_slopes'][cell]:.5f} |"
        )
    lines += [
        "",
        "| cell | " + " | ".join(f"seed {seed}" for seed in config["seeds"]) + " |",
        "|---|" + "---:|" * len(config["seeds"]),
    ]
    for cell in CELLS:
        values = summary["metrics"][cell]["values"]
        lines.append("| " + cell + " | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    lines += [
        "",
        "| pair | accuracy difference | 95% paired-bootstrap CI | McNemar p | Holm p |",
        "|---|---:|---|---:|---:|",
    ]
    for entry in summary["pairwise"]:
        low, high = entry["bootstrap_interval"]
        lines.append(
            f"| {entry['first']} - {entry['second']} | {entry['accuracy_difference']:+.4f} | "
            f"[{low:+.4f}, {high:+.4f}] | {entry['mcnemar_p']:.4g} | {entry['holm_p']:.4g} |"
        )
    lines += [
        "",
        "McNemar uses seed-0 predictions; the bootstrap interval uses seed-averaged",
        "per-example correctness. Neither accounts for seed variability, so the",
        "per-seed table above is the stricter check. A CI that excludes zero and a",
        "Holm-corrected McNemar p < 0.05 support a difference on this test set;",
        "neither establishes a universal ordering.",
        "",
        "Gradient slope: least-squares slope of log || d ||h_T||^2 / d x_t || against",
        f"lag on unpadded test sentences of {config['probe_length']} tokens, averaged over",
        "the probed seeds; exp(slope) is the average per-step retention.",
        "",
        *notes,
    ]
    (run_directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    args = argument_parser(__doc__).parse_args()
    config = select_preset(PRESETS, args.tier)
    if args.steps is not None:
        config["max_steps"] = args.steps

    set_seed(args.seed)
    device = get_device(args.device)
    print(environment_summary(str(device)))

    train_examples, validation_examples, test_examples = load_examples(config)
    tokenizer = WordTokenizer.from_text(
        " ".join(text for text, _ in train_examples),
        special_tokens=SPECIAL_TOKENS,
        min_freq=config["min_frequency"],
    )
    if tokenizer.pad_id is None:
        raise RuntimeError("tokenizer is missing <pad>")
    loaders = build_loaders(
        train_examples, validation_examples, test_examples, tokenizer, config
    )
    print(
        f"data: train={len(loaders[0].dataset)} validation={len(loaders[1].dataset)} "
        f"test={len(loaders[2].dataset)} vocabulary={tokenizer.vocab_size}"
    )

    run_directory = prepare_run(AREA, NAME, args.tier, args.output_directory)
    results: dict[str, list[dict]] = {cell: [] for cell in CELLS}
    for cell in CELLS:
        for index, seed in enumerate(config["seeds"]):
            checkpoint = (
                run_directory / "checkpoints" / f"{cell}.pt" if index == 0 else None
            )
            results[cell].append(
                train_one(
                    cell,
                    seed + args.seed,
                    tokenizer,
                    loaders,
                    config,
                    device,
                    checkpoint_path=checkpoint,
                )
            )

    summary = aggregate(results, config)
    write_config(
        run_directory,
        {
            **config,
            "tier": args.tier,
            "seed": args.seed,
            "device": str(device),
            "vocabulary_size": tokenizer.vocab_size,
            "train_examples": len(loaders[0].dataset),
            "validation_examples": len(loaders[1].dataset),
            "test_examples": len(loaders[2].dataset),
        },
    )
    write_metrics(run_directory, [], summary=summary)
    (run_directory / "raw_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    write_report(run_directory, summary, config, args.tier)

    # Learning curves (mean ± std across seeds).
    steps = [entry["step"] for entry in results[CELLS[0]][0]["history"]]
    curves = {}
    for cell in CELLS:
        values = np.asarray(
            [[entry["validation_loss"] for entry in run["history"]] for run in results[cell]]
        )
        curves[cell] = (steps, values.mean(axis=0).tolist(), values.std(axis=0).tolist())
    plot_lines(
        curves,
        run_directory / "validation_loss.png",
        xlabel="optimizer step",
        ylabel="validation cross-entropy",
        title="Sentiment: validation loss at matched parameters",
    )

    plot_bars(
        {
            cell: (
                summary["metrics"][cell]["accuracy_mean"],
                summary["metrics"][cell]["accuracy_std"],
            )
            for cell in CELLS
        },
        run_directory / "test_accuracy.png",
        ylabel="test accuracy",
        title="Sentiment: test accuracy (mean ± std over seeds)",
    )

    plot_gradient_decay(
        {cell: results[cell][0]["gradient_norms"] for cell in CELLS},
        run_directory / "gradient_decay.png",
    )

    series_by_length, counts, edges = accuracy_by_length(results, config)
    bucket_labels = [f"{edges[index]}-{edges[index + 1]}" for index in range(len(edges) - 1)]
    plot_lines(
        {
            cell: (list(range(len(values))), values, None)
            for cell, values in series_by_length.items()
        },
        run_directory / "accuracy_by_length.png",
        xlabel="token-length bucket (see report)",
        ylabel="test accuracy",
        title="Sentiment accuracy by sequence length",
        xticks=bucket_labels,
    )
    (run_directory / "length_buckets.json").write_text(
        json.dumps({"edges": edges, "counts": counts, "series": series_by_length}, indent=2),
        encoding="utf-8",
    )

    print("\n" + (run_directory / "report.md").read_text(encoding="utf-8"))
    print(f"artifacts: {run_directory}")


if __name__ == "__main__":
    main()
