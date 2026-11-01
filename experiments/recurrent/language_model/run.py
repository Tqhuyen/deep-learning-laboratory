"""Recurrent language model: RNN, LSTM or GRU, same data as the transformer run.

The point of this experiment is comparison. Same corpus, same tokenizer, same
training loop as ``language_modeling/character_gpt`` — only the sequence mixers
change. That isolates the architectural question instead of confounding it with
data or optimisation differences.

    python experiments/recurrent/language_model/run.py --tier tiny --cell lstm
    python experiments/recurrent/language_model/run.py --tier smoke --cell gru

The model class below lives in the experiment, not in ``laboratory``, to show
the intended workflow: the laboratory provides primitives, the experiment
composes them.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from laboratory import (
    GRU,
    LSTM,
    RNN,
    CharTokenizer,
    LanguageModelDataset,
    Trainer,
    cosine_warmup,
    count_parameters,
    generate,
    get_device,
    make_optimizer,
    perplexity,
    save_tokenizer,
    set_seed,
    token_cross_entropy,
    train_val_split,
)
from laboratory.experiment import (
    argument_parser,
    data_path,
    environment_summary,
    prepare_run,
    select_preset,
    write_config,
    write_metrics,
)

AREA = "recurrent"
NAME = "language_model"

RECURRENT_LAYERS = {"rnn": RNN, "lstm": LSTM, "gru": GRU}

PRESETS = {
    "tiny": dict(
        cell="lstm",
        block_size=32,
        batch_size=8,
        hidden_size=32,
        num_layers=1,
        dropout=0.0,
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup_steps=1,
        max_steps=5,
        log_every=1,
        validation_batches=2,
        sample_tokens=40,
    ),
    "smoke": dict(
        cell="lstm",
        block_size=64,
        batch_size=16,
        hidden_size=128,
        num_layers=2,
        dropout=0.0,
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup_steps=10,
        max_steps=100,
        log_every=25,
        validation_batches=4,
        sample_tokens=80,
    ),
    "development": dict(
        cell="lstm",
        block_size=128,
        batch_size=32,
        hidden_size=256,
        num_layers=2,
        dropout=0.1,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_steps=100,
        max_steps=1500,
        log_every=50,
        validation_batches=20,
        sample_tokens=300,
    ),
    "full": dict(
        cell="lstm",
        block_size=256,
        batch_size=64,
        hidden_size=512,
        num_layers=3,
        dropout=0.1,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_steps=500,
        max_steps=5000,
        log_every=100,
        validation_batches=50,
        sample_tokens=500,
    ),
}


class RecurrentLanguageModel(nn.Module):
    """Embedding -> recurrent stack -> linear next-token head."""

    def __init__(
        self,
        vocabulary_size: int,
        hidden_size: int,
        num_layers: int,
        cell: str,
        dropout: float = 0.0,
    ):
        super().__init__()
        if cell not in RECURRENT_LAYERS:
            raise ValueError(f"unknown cell {cell!r}; choose from {sorted(RECURRENT_LAYERS)}")
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.recurrent = RECURRENT_LAYERS[cell](
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.head = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor | None = None):
        outputs, _ = self.recurrent(self.embedding(inputs))
        logits = self.head(outputs)
        loss = None
        if targets is not None:
            loss = token_cross_entropy(logits, targets)
        return logits, loss

def recurrent_step(model: RecurrentLanguageModel, batch):
    inputs, targets = batch
    _, loss = model(inputs, targets)
    return loss, {"perplexity": perplexity(loss)}


def main() -> None:
    parser = argument_parser(__doc__)
    parser.add_argument(
        "--cell",
        choices=sorted(RECURRENT_LAYERS),
        default=None,
        help="override the preset's recurrent cell",
    )
    args = parser.parse_args()

    config = select_preset(PRESETS, args.tier)
    if args.cell is not None:
        config["cell"] = args.cell
    if args.steps is not None:
        config["max_steps"] = args.steps

    set_seed(args.seed)
    device = get_device(args.device)
    print(environment_summary(str(device)))

    text = data_path("tiny_shakespeare.txt").read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    training_ids, validation_ids = train_val_split(ids, val_fraction=0.1)

    train_dataset = LanguageModelDataset(training_ids, config["block_size"])
    validation_dataset = LanguageModelDataset(validation_ids, config["block_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["batch_size"],
        drop_last=False,
    )

    model = RecurrentLanguageModel(
        vocabulary_size=tokenizer.vocab_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        cell=config["cell"],
        dropout=config["dropout"],
    )
    print(
        f"model: {config['cell']} with {count_parameters(model):,} parameters, "
        f"{len(train_dataset):,} training windows"
    )

    optimizer = make_optimizer(
        model.parameters(),
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = cosine_warmup(optimizer, config["warmup_steps"], config["max_steps"])
    trainer = Trainer(model, optimizer, device=device, clip_grad=1.0, scheduler=scheduler)

    run_directory = prepare_run(AREA, NAME, args.tier, args.output_directory)
    write_config(
        run_directory,
        {
            **config,
            "tier": args.tier,
            "seed": args.seed,
            "device": str(device),
            "vocabulary_size": tokenizer.vocab_size,
            "parameters": count_parameters(model),
        },
    )

    history = trainer.fit(
        train_loader,
        recurrent_step,
        epochs=10_000,
        val_loader=validation_loader,
        log_every=config["log_every"],
        max_steps=config["max_steps"],
        validation_batches=config["validation_batches"],
        checkpoint_path=run_directory / "checkpoint.pt",
    )
    save_tokenizer(tokenizer, run_directory / "tokenizer.json")

    prompt = "ROMEO:"
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    sample = generate(
        model,
        prompt_ids,
        max_new_tokens=config["sample_tokens"],
        context_window=256,
        temperature=0.8,
    )
    sample_text = tokenizer.decode(sample[0].tolist())
    (run_directory / "sample.txt").write_text(sample_text, encoding="utf-8")

    write_metrics(
        run_directory,
        history,
        final_loss=history[-1]["train_loss"],
        final_validation_loss=history[-1].get("val_loss"),
    )

    print("\n--- sample ---")
    print(sample_text)
    print(f"\nartifacts: {run_directory}")


if __name__ == "__main__":
    main()
