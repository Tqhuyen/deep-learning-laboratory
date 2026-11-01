# Conventions

This file is the contract for the repository. It answers three questions:
where code goes, how much compute a run may use, and what "done" means for a
change.

## 1. Layout

```
laboratory/                 shared, importable primitives
experiments/<area>/<name>/  one directory per experiment, each with run.py
tests/unit/                 correctness of the primitives (CPU, seconds)
tests/smoke/                every experiment at the tiny tier (CPU, under a minute)
data/                       small committed corpora; nothing is downloaded at runtime
docs/                       this contract and the paper-to-code guide
```

Rules:

* `laboratory/` contains no experiment-specific code, no CLI, and no data
  loading of a particular corpus. If a piece is only useful for one experiment,
  it lives in that experiment's `run.py` (see `RecurrentLanguageModel` in
  `experiments/recurrent/language_model/run.py`).
* An experiment may import any part of `laboratory/` and compose it freely.
  This is the point of the monorepo: experiments explore *any* component while
  sharing tokenizers, batching, training and testing.
* Names are full words in `snake_case`: `positional_encoding.py`, not
  `pos_enc.py`. Standard architecture names from the literature (LSTM, RoPE,
  ALiBi, GQA) stay as they are, because renaming them would make the code harder
  to map to papers.

## 2. Tiers: how much compute a run may spend

Every `run.py` implements four presets. The tier is chosen with `--tier`.

| tier | budget | what it must prove |
|---|---|---|
| `tiny` | < 10 s on CPU, 5 optimizer steps | shapes, masks, one backward pass, artifacts written |
| `smoke` | < 60 s on CPU, ~100 steps | loss decreases, no NaNs, config saved |
| `development` | minutes on one GPU, 1k–3k steps | real learning signal while iterating |
| `full` | hours or days | numbers you would report |

Non-negotiable rules:

* `--tier tiny` must succeed on a CPU-only machine with the committed data.
* `tiny` evaluates at most two validation batches (`validation_batches`) so the
  test suite stays fast.
* A preset is a complete configuration. Do not rely on hidden defaults for
  anything that changes results; put batch size, sequence length, model width,
  depth, learning rate and step budget in the preset.
* `--steps` overrides the step budget for debugging. Do not use it to produce
  reported numbers.
* `--device auto` must resolve to CPU when CUDA is unavailable. Never hard-code
  `cuda`.

## 3. Experiment contract

Each `experiments/<area>/<name>/run.py`:

1. Defines `AREA`, `NAME` and `PRESETS` with all four tiers.
2. Uses `laboratory.experiment.argument_parser`, `select_preset`,
   `prepare_run`, `write_config` and `write_metrics`.
3. Writes to `runs/<area>/<name>/<tier>/`:
   * `config.json` — resolved config, seed, device, parameter count;
   * `checkpoint.pt` — model (and optimizer) state;
   * `metrics.json` — per-epoch history and final metrics;
   * optional samples (`sample.txt`).
4. Seeds Python, NumPy and PyTorch with `--seed` before building data.
5. Uses `num_workers=0` (or documents another choice) so results are
   reproducible on every platform.
6. Runs a small end-of-training evaluation that is meaningful for the task
   (perplexity, token accuracy, exact match), not just "it did not crash".

Add an experiment by copying the shape of an existing one:

```text
experiments/attention/sliding_window/
├── run.py
└── README.md          # optional: what question this run answers
```

## 4. Scaling: how to grow a run without fooling yourself

Compute is roughly

```text
training_tokens = batch_size × sequence_length × optimizer_steps
```

and that is the number that makes two runs comparable, not "epochs".

Scaling playbook:

1. **Freeze the comparison first.** Fix data, tokenizer, step budget and
   evaluation. Then change one axis.
2. **Grow width before depth for small models.** Parameters scale with
   `d_model² × layers`. Keep `head_dim = d_model / n_heads` at 32–64 and grow
   `d_model` and `n_heads` together.
3. **Sequence length is the expensive axis for attention** (quadratic memory
   and time) and linear for recurrence. Do not change it while comparing
   architectures unless the paper's claim is about context length.
4. **Batch size and learning rate move together.** `3e-4` AdamW is a sane
   default at `d_model=256`; smaller models tolerate `1e-3`, larger models need
   less. Warm up for 1–5 % of steps.
5. **When memory runs out, accumulate gradients**, do not silently shrink the
   logical batch size. The effective batch is
   `micro_batch × accumulation × devices`.
6. **Log tokens seen, parameters and wall time** alongside loss. Two runs with
   different budgets are not a comparison.
7. **Report at least three seeds** for any claim, as mean ± standard deviation,
   plus the exact preset and commit.

Hardware-limited strategy:

* Character-level corpora keep the vocabulary tiny (65 tokens here), which
  makes models small and runs fast without hiding the mechanics.
* Train on a contiguous slice of the corpus when the full set is too slow; note
  the slice in `config.json`.
* Prefer `tiny`/`smoke` for architectural debugging. Use `development` only to
  confirm a hypothesis.
* On CPU, reduce `batch_size` first, then `block_size`, then model width.

## 5. Testing

Two layers, both CPU-only and offline:

**Unit tests** (`tests/unit/`) check claims that can be proven:

* analytical equivalence: LayerNorm against `F.layer_norm`, LSTM/GRU/RNN
  against PyTorch's fused kernels with copied weights, attention against the
  hand-written formula;
* structural invariants: causality (changing a future token must not change
  past logits), padding independence, RoPE dot products depending only on
  relative position, norm preservation;
* contracts: shapes, mask conventions, error paths, checkpoint round-trips.

They must stay fast (the whole suite is under a minute on CPU including the
smoke runs) and must not download data. Do not test wiring or restate the
implementation in a test; if a test would pass with the maths broken, delete it.

**Smoke tests** (`tests/smoke/`) discover every `experiments/**/run.py` and run
it with `--tier tiny --device cpu`, asserting exit code and artifacts. If an
experiment can no longer run on a laptop, CI fails here.

Run:

```bash
pytest -q                 # everything
pytest tests/unit -q      # maths and contracts only
```

```bash
# one experiment, all tiers
python experiments/language_modeling/character_gpt/run.py --tier tiny
python experiments/language_modeling/character_gpt/run.py --tier smoke
python experiments/language_modeling/character_gpt/run.py --tier development
```

## 6. Reproducibility

* Every run saves its resolved config, seed, device and parameter count.
* `environment_summary()` prints library and hardware versions at start-up.
* Checkpoints include model, optimizer and metadata; do not reuse a checkpoint
  across architectures.
* Keep `data/` small and committed. If a paper's dataset is too large, write a
  preparation script and document the hash — never make `run.py` download
  silently.

## 7. Where a paper's idea usually lands

| paper component | file |
|---|---|
| tokenizer / vocabulary | `laboratory/tokenization.py` |
| batching, teacher forcing, masking of pads | `laboratory/data.py` |
| positional information (absolute, rotary, biases) | `laboratory/positional_encoding.py` |
| attention (heads, grouping, cross-attention, masks) | `laboratory/attention.py` |
| normalisation, activation, feed-forward variants | `laboratory/layers.py` |
| block structure, residual order, model assembly | `laboratory/transformer.py` |
| recurrent cells | `laboratory/recurrent.py` |
| optimiser, schedule, loop, checkpointing | `laboratory/training.py` |

`docs/paper-to-code.md` turns this table into a step-by-step workflow with
worked examples.
