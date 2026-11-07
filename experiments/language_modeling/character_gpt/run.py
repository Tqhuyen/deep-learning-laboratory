"""Character-level language model with a decoder-only transformer.

Pipeline: text -> :class:`CharTokenizer` -> :class:`LanguageModelDataset` ->
:class:`GPT` (learned positions, causal self-attention) -> next-token
cross-entropy -> sampling.

The prompt is Shakespeare; the model is small enough to run on a laptop and
the ``development`` tier produces recognisable text on a single GPU.

    python experiments/language_modeling/character_gpt/run.py --tier tiny
    python experiments/language_modeling/character_gpt/run.py --tier smoke
    python experiments/language_modeling/character_gpt/run.py --tier development

Artifacts land in ``runs/language_modeling/character_gpt/<tier>/``.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from laboratory import (
    CharTokenizer,
    GPT,
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

AREA = "language_modeling"
NAME = "character_gpt"

PRESETS = {
    "tiny": dict(
        block_size=32,
        batch_size=8,
        d_model=64,
        n_layers=2,
        n_heads=2,
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
        block_size=64,
        batch_size=16,
        d_model=128,
        n_layers=2,
        n_heads=4,
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
        block_size=128,
        batch_size=32,
        d_model=256,
        n_layers=4,
        n_heads=8,
        dropout=0.0,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_steps=100,
        max_steps=2000,
        log_every=50,
        validation_batches=20,
        sample_tokens=300,
    ),
    "full": dict(
        block_size=256,
        batch_size=64,
        d_model=512,
        n_layers=8,
        n_heads=8,
        dropout=0.1,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_steps=500,
        max_steps=8000,
        log_every=100,
        validation_batches=50,
        sample_tokens=500,
    ),
}


def language_model_step(model: GPT, batch) -> tuple[torch.Tensor, dict[str, float]]:
    """One training step: predict the next character at every position."""
    inputs, targets = batch
    _, loss = model(inputs, targets)
    return loss, {"perplexity": perplexity(loss)}


def main() -> None:
    args = argument_parser(__doc__).parse_args()
    config = select_preset(PRESETS, args.tier)
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

    model = GPT(
        vocab_size=tokenizer.vocab_size,
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        max_len=config["block_size"],
        dropout=config["dropout"],
        pos="learned",
    )
    print(
        f"model: {count_parameters(model):,} parameters, "
        f"{tokenizer.vocab_size} characters, {len(train_dataset):,} training windows"
    )

    optimizer = make_optimizer(
        model,
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
        language_model_step,
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
        temperature=0.8,
        top_k=40,
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
