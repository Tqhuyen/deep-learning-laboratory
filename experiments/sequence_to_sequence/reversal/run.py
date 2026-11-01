"""Sequence-to-sequence reversal with an encoder-decoder transformer.

A deliberately synthetic translation task: the target is the source sequence
reversed. It is verifiable (exact match) and cheap, while exercising exactly the
machinery a translation paper needs:

text -> :class:`WordTokenizer` with ``<pad> <bos> <eos> <unk>`` ->
:class:`Seq2SeqDataset` -> :class:`Seq2SeqCollator` (teacher forcing) ->
:class:`Seq2SeqTransformer` (causal decoder + cross-attention) -> greedy
decoding -> exact-match accuracy.

    python experiments/sequence_to_sequence/reversal/run.py --tier tiny
    python experiments/sequence_to_sequence/reversal/run.py --tier smoke

Swap ``make_pairs`` for a real parallel corpus (or load a TSV) and nothing else
in this file needs to change.
"""

from __future__ import annotations

import random

import torch
from torch.utils.data import DataLoader

from laboratory import (
    Seq2SeqCollator,
    Seq2SeqDataset,
    Seq2SeqTransformer,
    Trainer,
    WordTokenizer,
    accuracy,
    cosine_warmup,
    count_parameters,
    get_device,
    lengths_mask,
    pad_sequences,
    save_tokenizer,
    set_seed,
    token_cross_entropy,
)
from laboratory.experiment import (
    argument_parser,
    environment_summary,
    prepare_run,
    select_preset,
    write_config,
    write_metrics,
)

AREA = "sequence_to_sequence"
NAME = "reversal"

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")

PRESETS = {
    "tiny": dict(
        train_pairs=64,
        validation_pairs=16,
        max_len=16,
        batch_size=4,
        d_model=32,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup_steps=1,
        max_steps=5,
        log_every=1,
        validation_batches=2,
        sample_count=3,
    ),
    "smoke": dict(
        train_pairs=512,
        validation_pairs=64,
        max_len=16,
        batch_size=16,
        d_model=64,
        n_layers=2,
        n_heads=2,
        dropout=0.0,
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup_steps=20,
        max_steps=150,
        log_every=25,
        validation_batches=4,
        sample_count=5,
    ),
    "development": dict(
        train_pairs=20_000,
        validation_pairs=512,
        max_len=24,
        batch_size=64,
        d_model=128,
        n_layers=3,
        n_heads=4,
        dropout=0.1,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_steps=200,
        max_steps=2000,
        log_every=100,
        validation_batches=10,
        sample_count=8,
    ),
    "full": dict(
        train_pairs=100_000,
        validation_pairs=2000,
        max_len=32,
        batch_size=128,
        d_model=256,
        n_layers=4,
        n_heads=8,
        dropout=0.1,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_steps=500,
        max_steps=10_000,
        log_every=100,
        validation_batches=20,
        sample_count=10,
    ),
}


def make_pairs(count: int, seed: int, minimum_length: int = 3, maximum_length: int = 8):
    """Synthetic parallel corpus: ``"3 8 1" -> "1 8 3"``."""
    generator = random.Random(seed)
    pairs = []
    for _ in range(count):
        length = generator.randint(minimum_length, maximum_length)
        digits = [str(generator.randint(0, 9)) for _ in range(length)]
        pairs.append((" ".join(digits), " ".join(reversed(digits))))
    return pairs


def build_step(pad_id: int):
    """Teacher-forced cross-entropy plus token accuracy."""

    def step(model: Seq2SeqTransformer, batch) -> tuple[torch.Tensor, dict[str, float]]:
        logits = model(
            batch["src"],
            batch["tgt_in"],
            src_key_padding_mask=batch["src_mask"],
        )
        loss = token_cross_entropy(logits, batch["tgt_out"], ignore_index=pad_id)
        return loss, {
            "token_accuracy": accuracy(
                logits.detach(), batch["tgt_out"], ignore_index=pad_id
            )
        }

    return step


@torch.no_grad()
def greedy_decode(
    model: Seq2SeqTransformer,
    tokenizer: WordTokenizer,
    sources: list[str],
    device: torch.device,
    max_new_tokens: int,
) -> list[list[int]]:
    """Run the model on raw strings and return token ids up to ``<eos>``."""
    encoded = [tokenizer.encode(source) for source in sources]
    padded, lengths = pad_sequences(encoded, tokenizer.pad_id)
    mask = lengths_mask(lengths, padded.size(1)).to(device)
    generated = model.generate(
        padded.to(device), max_new_tokens=max_new_tokens, src_key_padding_mask=mask
    )
    results = []
    for row in generated.tolist():
        tokens = []
        for token_id in row[1:]:  # skip <bos>
            if token_id == tokenizer.eos_id:
                break
            tokens.append(token_id)
        results.append(tokens)
    return results


def main() -> None:
    args = argument_parser(__doc__).parse_args()
    config = select_preset(PRESETS, args.tier)
    if args.steps is not None:
        config["max_steps"] = args.steps

    set_seed(args.seed)
    device = get_device(args.device)
    print(environment_summary(str(device)))

    tokenizer = WordTokenizer.from_text(
        " ".join(str(digit) for digit in range(10)),
        special_tokens=SPECIAL_TOKENS,
    )
    pad_id, bos_id, eos_id = tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id
    assert pad_id is not None and bos_id is not None and eos_id is not None

    train_pairs = make_pairs(config["train_pairs"], seed=args.seed)
    validation_pairs = make_pairs(
        config["validation_pairs"], seed=args.seed + 1, minimum_length=3, maximum_length=6
    )

    train_dataset = Seq2SeqDataset(
        train_pairs, tokenizer, max_src_len=config["max_len"], max_tgt_len=config["max_len"]
    )
    validation_dataset = Seq2SeqDataset(
        validation_pairs,
        tokenizer,
        max_src_len=config["max_len"],
        max_tgt_len=config["max_len"],
    )
    collator = Seq2SeqCollator(
        pad_id,
        bos_id,
        eos_id,
        max_src_len=config["max_len"],
        max_tgt_len=config["max_len"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["batch_size"],
        drop_last=False,
        collate_fn=collator,
    )

    model = Seq2SeqTransformer(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        # +1 for the decoder's <bos>, +1 for <eos>; targets are capped at max_len.
        max_len=config["max_len"] + 2,
        dropout=config["dropout"],
        pos="sinusoidal",
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
    )
    print(f"model: {count_parameters(model):,} parameters, vocabulary={tokenizer.vocab_size}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
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
        build_step(pad_id),
        epochs=10_000,
        val_loader=validation_loader,
        log_every=config["log_every"],
        max_steps=config["max_steps"],
        validation_batches=config["validation_batches"],
        checkpoint_path=run_directory / "checkpoint.pt",
    )
    save_tokenizer(tokenizer, run_directory / "tokenizer.json")

    sample_pairs = validation_pairs[: config["sample_count"]]
    predictions = greedy_decode(
        model,
        tokenizer,
        [source for source, _ in sample_pairs],
        device,
        max_new_tokens=config["max_len"] + 1,
    )
    exact_matches = 0
    print("\n--- greedy decoding ---")
    for (source, target), predicted_ids in zip(sample_pairs, predictions):
        predicted = tokenizer.decode(predicted_ids)
        exact_matches += int(predicted_ids == tokenizer.encode(target))
        print(f"source={source!r}  target={target!r}  predicted={predicted!r}")
    exact_match_rate = exact_matches / max(len(sample_pairs), 1)
    print(f"sample exact-match rate: {exact_match_rate:.2%}")

    write_metrics(
        run_directory,
        history,
        final_loss=history[-1]["train_loss"],
        sample_exact_match_rate=exact_match_rate,
    )
    print(f"\nartifacts: {run_directory}")


if __name__ == "__main__":
    main()
