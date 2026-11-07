"""Recurrent language modelling: RNN vs GRU vs LSTM at matched parameters.

Question: at the same parameter budget and token budget, which cell predicts the
next character best, and does the measured gradient reach explain the ordering?

Design:

* Tiny Shakespeare, character-level (vocabulary 65), contiguous validation tail.
* Parameters matched with :func:`laboratory.match_language_model_hidden_size`;
  per-layer cell parameters are ``m * (I*H + H^2 + 2H)`` with ``m = 1, 3, 4``.
* Identical optimiser, schedule, batch size and optimizer steps; three seeds in
  ``development``, five in ``full``.
* Reported metric: bits per character (BPC = cross-entropy / ln 2). Paired
  bootstrap over validation windows, Holm-corrected across the three pairs.
* Mechanism: ``|| d ||h_T||^2 / d x_t ||`` for every input position ``t`` of
  unpadded validation windows (how strongly the final state still depends on
  each earlier character), plus samples generated from the same prompt and seed.

Run:

    python experiments/language_modeling/recurrent_generation/run.py --tier tiny
    python experiments/language_modeling/recurrent_generation/run.py --tier development
"""

from __future__ import annotations

import itertools
import json
import math
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
    Trainer,
    cosine_warmup,
    count_parameters,
    decay_slope,
    generate,
    get_device,
    holm_adjust,
    make_optimizer,
    match_language_model_hidden_size,
    paired_bootstrap_difference,
    paired_bootstrap_p_value,
    save_checkpoint,
    set_seed,
    timestep_gradient_norms,
    train_val_split,
)
from laboratory.evaluation.plots import plot_bars, plot_gradient_decay, plot_lines
from laboratory.experiment import (
    argument_parser,
    data_path,
    environment_summary,
    prepare_run,
    select_preset,
    write_config,
    write_metrics,
)

AREA = "language_modeling"
NAME = "recurrent_generation"

CELLS = ("rnn", "gru", "lstm")
PROMPT = "ROMEO:"

PRESETS = {
    "tiny": dict(
        max_steps=5,
        batch_size=8,
        block_size=32,
        embedding_dim=32,
        target_parameters=100_000,
        learning_rate=1e-3,
        weight_decay=0.0,
        seeds=(0,),
        eval_every=2,
        validation_batches=1,
        sample_tokens=40,
        final_batches=2,
        bootstrap_repetitions=400,
    ),
    "smoke": dict(
        max_steps=100,
        batch_size=16,
        block_size=64,
        embedding_dim=64,
        target_parameters=400_000,
        learning_rate=1e-3,
        weight_decay=0.0,
        seeds=(0,),
        eval_every=20,
        validation_batches=4,
        sample_tokens=80,
        final_batches=8,
        bootstrap_repetitions=1000,
    ),
    "development": dict(
        max_steps=2000,
        batch_size=32,
        block_size=128,
        embedding_dim=64,
        target_parameters=1_500_000,
        learning_rate=3e-4,
        weight_decay=0.01,
        seeds=(0, 1, 2),
        eval_every=100,
        validation_batches=10,
        sample_tokens=200,
        final_batches=40,
        bootstrap_repetitions=2000,
    ),
    "full": dict(
        max_steps=6000,
        batch_size=64,
        block_size=256,
        embedding_dim=128,
        target_parameters=3_000_000,
        learning_rate=3e-4,
        weight_decay=0.01,
        seeds=(0, 1, 2, 3, 4),
        eval_every=250,
        validation_batches=20,
        sample_tokens=400,
        final_batches=80,
        bootstrap_repetitions=3000,
    ),
}


def make_loaders(tokenizer, config):
    text = data_path("tiny_shakespeare.txt").read_text(encoding="utf-8")
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    training_ids, validation_ids = train_val_split(ids, val_fraction=0.1)
    train = LanguageModelDataset(training_ids, config["block_size"])
    validation = LanguageModelDataset(validation_ids, config["block_size"])
    train_loader = DataLoader(
        train,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
    )
    return train_loader, validation_loader, train, validation


def make_step():
    def step(model, batch):
        inputs, targets = batch
        _, loss = model(inputs, targets)
        return loss, {"bpc": loss.detach().item() / math.log(2.0)}

    return step


@torch.no_grad()
def per_window_nll(model, loader, device, max_batches=None):
    """Mean NLL per validation window, shape ``(windows,)``."""
    model.eval()
    values = []
    for index, (inputs, targets) in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)[0]
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).view(targets.size(0), targets.size(1))
        values.append(token_losses.mean(dim=1).cpu().numpy())
    return np.concatenate(values)


def train_one(cell, seed, tokenizer, loaders, config, device, checkpoint_path=None):
    train_loader, validation_loader, _, _ = loaders
    set_seed(seed)

    hidden_size, expected_parameters = match_language_model_hidden_size(
        cell,
        target_parameters=config["target_parameters"],
        vocabulary_size=tokenizer.vocab_size,
        embedding_dim=config["embedding_dim"],
    )
    model = RecurrentLanguageModel(
        vocabulary_size=tokenizer.vocab_size,
        embedding_dim=config["embedding_dim"],
        hidden_size=hidden_size,
        num_layers=1,
        cell=cell,
    )
    actual_parameters = count_parameters(model)
    if abs(actual_parameters - expected_parameters) > 0:
        raise RuntimeError(
            f"capacity formula mismatch for {cell}: {actual_parameters} vs {expected_parameters}"
        )

    optimizer = make_optimizer(
        model,
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
                    "train_bpc": logs["bpc"],
                    "validation_bpc": validation["bpc"],
                }
            )
            print(
                f"{cell} seed={seed} step={step} "
                f"train_bpc={logs['bpc']:.4f} "
                f"val_bpc={validation['bpc']:.4f}"
            )

    window_nll = per_window_nll(
        model, validation_loader, device, max_batches=config["final_batches"]
    )
    final_bpc = float(window_nll.mean() / math.log(2.0))
    print(f"{cell} seed={seed} validation_bpc={final_bpc:.4f}")

    torch.manual_seed(seed + 991)
    prompt_ids = torch.tensor([tokenizer.encode(PROMPT)], device=device)
    generated = generate(
        model,
        prompt_ids,
        max_new_tokens=config["sample_tokens"],
        context_window=config["block_size"],
        temperature=0.8,
        top_k=40,
    )
    sample = tokenizer.decode(generated[0].tolist())

    probe_inputs, _ = next(iter(validation_loader))
    probe_inputs = probe_inputs.to(device)
    model.eval()
    embeddings = model.embedding(probe_inputs)
    gradient_norms = timestep_gradient_norms(model.recurrent, embeddings).numpy()

    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path, model, cell=cell, seed=seed)

    return {
        "cell": cell,
        "seed": seed,
        "hidden_size": hidden_size,
        "parameters": actual_parameters,
        "history": history,
        "validation_bpc": final_bpc,
        "window_nll": window_nll.tolist(),
        "gradient_norms": gradient_norms.tolist(),
        "gradient_slope": decay_slope(gradient_norms),
        "sample": sample,
    }


def aggregate(results, config):
    summary = {"parameters": {}, "metrics": {}, "pairwise": [], "gradient_slopes": {}}
    for cell in CELLS:
        runs = results[cell]
        summary["parameters"][cell] = {
            "hidden_size": runs[0]["hidden_size"],
            "parameters": runs[0]["parameters"],
        }
        bpcs = [run["validation_bpc"] for run in runs]
        summary["metrics"][cell] = {
            "bpc_mean": float(np.mean(bpcs)),
            "bpc_std": float(np.std(bpcs, ddof=1)) if len(bpcs) > 1 else 0.0,
            "values": [float(value) for value in bpcs],
        }
        summary["gradient_slopes"][cell] = float(
            np.mean([run["gradient_slope"] for run in runs])
        )

    pairs = [("rnn", "gru"), ("rnn", "lstm"), ("gru", "lstm")]
    raw_p_values = []
    for first, second in pairs:
        first_nll = np.mean([run["window_nll"] for run in results[first]], axis=0)
        second_nll = np.mean([run["window_nll"] for run in results[second]], axis=0)
        scale = math.log(2.0)
        interval = paired_bootstrap_difference(
            first_nll / scale,
            second_nll / scale,
            repetitions=config["bootstrap_repetitions"],
            seed=0,
        )
        p_value = paired_bootstrap_p_value(
            first_nll / scale,
            second_nll / scale,
            repetitions=config["bootstrap_repetitions"],
            seed=0,
        )
        raw_p_values.append(p_value)
        summary["pairwise"].append(
            {
                "first": first,
                "second": second,
                "bpc_difference": float((first_nll.mean() - second_nll.mean()) / scale),
                "bootstrap_interval": list(interval),
                "bootstrap_p": p_value,
            }
        )
    adjusted = holm_adjust(raw_p_values)
    for entry, corrected in zip(summary["pairwise"], adjusted):
        entry["holm_p"] = corrected
    return summary


def write_report(run_directory, summary, config, tier, notes=()):
    lines = [
        "# Recurrent language modelling: RNN vs GRU vs LSTM",
        "",
        f"Tier: `{tier}`. Seeds: {config['seeds']}.",
        f"Training tokens per run: {config['max_steps'] * config['batch_size'] * config['block_size']:,}.",
        "",
        "Parameter matching: per-layer cell parameters are",
        "`m * (I*H + H^2 + 2H)` with `m = 1` (RNN), `3` (GRU), `4` (LSTM);",
        "hidden size is solved for the shared budget. BPC = CE / ln 2.",
        "",
        "| cell | hidden | parameters | validation BPC (mean ± std) | gradient slope |",
        "|---|---:|---:|---:|---:|",
    ]
    for cell in CELLS:
        parameters = summary["parameters"][cell]
        metrics = summary["metrics"][cell]
        lines.append(
            f"| {cell} | {parameters['hidden_size']} | {parameters['parameters']:,} | "
            f"{metrics['bpc_mean']:.4f} ± {metrics['bpc_std']:.4f} | "
            f"{summary['gradient_slopes'][cell]:.5f} |"
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
        "| pair | Δ BPC | 95% paired-bootstrap CI | bootstrap p | Holm p |",
        "|---|---:|---|---:|---:|",
    ]
    for entry in summary["pairwise"]:
        low, high = entry["bootstrap_interval"]
        lines.append(
            f"| {entry['first']} - {entry['second']} | {entry['bpc_difference']:+.4f} | "
            f"[{low:+.4f}, {high:+.4f}] | {entry['bootstrap_p']:.4g} | {entry['holm_p']:.4g} |"
        )
    lines += [
        "",
        "A negative Δ BPC means the first cell has lower bits per character.",
        "The bootstrap resamples validation windows, so its interval can be much",
        "narrower than run-to-run spread; the per-seed table is the stricter check.",
        "CIs quantify uncertainty on this corpus, not a universal ordering.",
        "",
        "Samples are in `samples.txt` (same prompt, same sampling seed per cell).",
        "",
        "Gradient slope: least-squares slope of log || d ||h_T||^2 / d x_t || against",
        f"lag on unpadded validation windows of {config['block_size']} characters, averaged",
        "over the probed seeds; exp(slope) is the average per-step retention.",
        "",
        *notes,
    ]
    (run_directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
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

    text = data_path("tiny_shakespeare.txt").read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    loaders = make_loaders(tokenizer, config)
    print(
        f"data: train_windows={len(loaders[2]):,} validation_windows={len(loaders[3]):,} "
        f"vocabulary={tokenizer.vocab_size}"
    )

    run_directory = prepare_run(AREA, NAME, args.tier, args.output_directory)
    results = {cell: [] for cell in CELLS}
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
            "train_windows": len(loaders[2]),
            "validation_windows": len(loaders[3]),
        },
    )
    write_metrics(run_directory, [], summary=summary)
    (run_directory / "raw_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    write_report(run_directory, summary, config, args.tier)

    steps = [entry["step"] for entry in results[CELLS[0]][0]["history"]]
    curves = {}
    for cell in CELLS:
        values = np.asarray(
            [[entry["validation_bpc"] for entry in run["history"]] for run in results[cell]]
        )
        curves[cell] = (steps, values.mean(axis=0).tolist(), values.std(axis=0).tolist())
    plot_lines(
        curves,
        run_directory / "validation_bpc.png",
        xlabel="optimizer step",
        ylabel="validation bits per character",
        title="Language modelling: validation BPC at matched parameters",
    )

    plot_bars(
        {
            cell: (
                summary["metrics"][cell]["bpc_mean"],
                summary["metrics"][cell]["bpc_std"],
            )
            for cell in CELLS
        },
        run_directory / "final_bpc.png",
        ylabel="validation bits per character",
        title="Language modelling: final BPC (mean ± std over seeds)",
    )

    plot_gradient_decay(
        {cell: results[cell][0]["gradient_norms"] for cell in CELLS},
        run_directory / "gradient_decay.png",
    )

    samples = ["# Samples (same prompt and sampling seed per cell)", ""]
    for cell in CELLS:
        samples += [f"## {cell}", "", results[cell][0]["sample"], ""]
    (run_directory / "samples.txt").write_text("\n".join(samples), encoding="utf-8")

    print("\n" + (run_directory / "report.md").read_text(encoding="utf-8"))
    print(f"artifacts: {run_directory}")


if __name__ == "__main__":
    main()
