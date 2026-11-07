"""Context-length sweep for the trained recurrent language models.

The development checkpoints were trained with 128 characters of context. This
script evaluates the same checkpoints on validation windows of 32, 64, 128 and
256 characters. It is a regime check, not a proof: a model trained at 128 is
in-distribution below that length and extrapolating above it.

Each context length uses the first ``batches x batch_size`` non-overlapping
validation windows, so longer contexts cover more text. Within one context the
three cells see identical characters; across contexts both the length and the
text change. The defaults reproduce ``results/recurrent_comparison/contexts``.

    python analysis/context_sweep.py \
        --run-directory runs/language_modeling/recurrent_generation/development \
        --output-directory results/recurrent_comparison/contexts
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from laboratory import (
    CharTokenizer,
    LanguageModelDataset,
    RecurrentLanguageModel,
    count_parameters,
    get_device,
    match_language_model_hidden_size,
    train_val_split,
)
from laboratory.experiment import data_path

CELLS = ("rnn", "gru", "lstm")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    args = parse_arguments()
    device = get_device(args.device)
    config = json.loads((args.run_directory / "config.json").read_text(encoding="utf-8"))

    text = data_path("tiny_shakespeare.txt").read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    _, validation_ids = train_val_split(ids, val_fraction=0.1)

    results: dict[str, dict[str, float]] = {cell: {} for cell in CELLS}
    characters: dict[str, int] = {}
    for context in args.contexts:
        dataset = LanguageModelDataset(validation_ids, context)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        batches = []
        for index, batch in enumerate(loader):
            if index >= args.batches:
                break
            batches.append((batch[0].to(device), batch[1].to(device)))

        for cell in CELLS:
            checkpoint_path = args.run_directory / "checkpoints" / f"{cell}.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            hidden, _ = match_language_model_hidden_size(
                cell,
                target_parameters=config["target_parameters"],
                vocabulary_size=tokenizer.vocab_size,
                embedding_dim=config["embedding_dim"],
            )
            model = RecurrentLanguageModel(
                vocabulary_size=tokenizer.vocab_size,
                embedding_dim=config["embedding_dim"],
                hidden_size=hidden,
                cell=cell,
            ).to(device)
            model.load_state_dict(checkpoint["model"])
            model.eval()

            total_loss, total_tokens = 0.0, 0
            with torch.no_grad():
                for inputs, targets in batches:
                    logits = model(inputs)[0]
                    total_loss += F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                        reduction="sum",
                    ).item()
                    total_tokens += targets.numel()
            bpc = total_loss / total_tokens / math.log(2.0)
            results[cell][str(context)] = bpc
            characters[str(context)] = total_tokens
            print(
                f"{cell} context={context} hidden={hidden} "
                f"parameters={count_parameters(model):,} bpc={bpc:.4f}"
            )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "context_sweep.json").write_text(
        json.dumps(
            {
                "trained_context": config["block_size"],
                "contexts": args.contexts,
                "batches": args.batches,
                "batch_size": args.batch_size,
                "evaluated_characters": characters,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    for cell in CELLS:
        values = [results[cell][str(context)] for context in args.contexts]
        axis.plot(args.contexts, values, marker="o", linewidth=1.6, label=cell)
    axis.set_xlabel("evaluation context length (characters)")
    axis.set_ylabel("validation bits per character")
    axis.set_title(
        f"Context sweep, checkpoints trained at {config['block_size']} characters"
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(args.output_directory / "context_sweep.png", dpi=160)
    plt.close(figure)
    print(f"wrote {args.output_directory}")


if __name__ == "__main__":
    main()
