# From a paper to a runnable experiment

This is the workflow the repository is built around. It assumes you have read a
paper and want to know whether its idea survives contact with code, on your
machine, without reimplementing a whole codebase.

## 1. Locate the change

Papers modify a small number of things. Write down which one applies before
opening an editor:

| claim in the paper | where it lands | interface to keep |
|---|---|---|
| "we change how positions are encoded" | `laboratory/blocks/positional_encoding.py` or `attn_mask`/bias in `laboratory/blocks/attention.py` | `(B, T, D) -> (B, T, D)` |
| "we change the attention pattern" | `laboratory/blocks/attention.py` | `MultiHeadAttention.forward(x, memory, attn_mask, key_padding_mask, is_causal)` |
| "we change the feed-forward block" | `laboratory/blocks/feedforward.py` | `(B, T, D) -> (B, T, D)` |
| "we change the residual/normalisation order" | `EncoderLayer`/`DecoderLayer` in `laboratory/models/transformer.py` | block keeps `forward(x, key_padding_mask)` |
| "we change the objective" | `laboratory/losses/<task>.py` (the experiment selects it) | `(predictions, targets) -> scalar` |
| "we change the data pipeline" | `laboratory/datasets/` and `laboratory/preprocessing/` | dataset item and collated batch stay the same |
| "we change optimisation" | `laboratory/optimization/`, `laboratory/experiment.py` presets | `Trainer` unchanged |

If a paper does not fit this table, it is probably changing several components;
still apply the changes one at a time.

## 2. Write the module with shapes first

Before writing maths, write the docstring: input shape, output shape, parameter
count formula, and the equation. Then implement, keeping the interface of the
thing you are replacing. That is what lets you drop it into an existing model
and compare against the baseline.

```python
class SlidingWindowAttention(nn.Module):
    """Restrict each query to the previous `window` keys.

    x: (batch, seq, d_model) -> (batch, seq, d_model)
    Mask: True where key j is visible to query i, i.e. 0 <= i - j < window.
    """
```

## 3. Write the test that would catch a wrong implementation

For a new attention pattern, the test is not "the output shape is right". It is
a property of the paper:

* causal or windowed: changing a token outside the window must not change the
  output;
* relative positions: shifting both query and key by the same offset must not
  change their score;
* normalisation: mean/variance or norm after the layer must match the stated
  formula;
* drop-in replacement: with the paper's hyperparameters set to the baseline's,
  the new module should be numerically close to the baseline when the paper
  says it is a generalisation.

```python
def test_window_blocks_the_distant_past():
    attention = MultiHeadAttention(d_model=16, n_heads=2, dropout=0.0)
    x = torch.randn(1, 8, 16)
    mask = torch.ones(8, 8, dtype=torch.bool).tril().triu(diagonal=-3)  # window = 4
    with torch.no_grad():
        original = attention(x, attn_mask=mask)
        modified_x = x.clone()
        modified_x[:, 0] += 10.0          # outside the last query's window
        modified = attention(modified_x, attn_mask=mask)
    assert torch.allclose(original[:, -1], modified[:, -1], atol=1e-6)
```

Run the test before the experiment. A failing maths test is cheap; a failing
training run is not.

## 4. Add a tier and run `tiny`

Add or edit a preset in the experiment's `PRESETS`, keeping the tier table in
`docs/conventions.md` in mind (tiny must finish in seconds on CPU). Then:

```bash
pytest tests/unit -q
python experiments/<area>/<name>/run.py --tier tiny
```

At this tier you are checking plumbing, not learning: does the forward pass run,
are masks right, does the checkpoint save, is the config recorded.

## 5. Confirm a learning signal at `smoke`

If the change is supposed to help, `smoke` must show the loss moving in the
right direction. If it does not, do not scale up. The usual causes, in order:

1. mask convention inverted (True means *keep*, everywhere in this repository);
2. wrong tensor layout when splitting heads (`(B, T, H, D)` vs `(B, H, T, D)`);
3. forgetting `is_causal=True` in a decoder, or passing it to cross-attention;
4. sequence length exceeding the positional table (`max_len`);
5. targets shifted twice (the dataset already shifts by one);
6. a scheduler stepped once per epoch instead of per step, or not at all.

## 6. Compare honestly

A result is a tuple: `(metric, preset, seed, parameters, tokens seen, commit)`.
Change one variable per run. Report at least three seeds. When the baseline and
the new method have different parameter counts, say so; if the paper's claim is
compute-matched, match tokens seen, not steps.

---

# Worked examples

## Example A: sliding-window attention (Mistral-style)

No new module is needed: the mask defines the pattern.

```python
def sliding_window_mask(seq_len: int, window: int) -> torch.Tensor:
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool).tril()
    return mask.triu(diagonal=-(window - 1))
```

```python
mask = sliding_window_mask(seq_len=x.size(1), window=64)
hidden = attention(x, attn_mask=mask)
```

To make it a real experiment, add a preset with `window` in the config, pass the
mask through `EncoderLayer` (add a `window` argument), and assert with a unit
test that tokens outside the window do not interact. Keep the full-attention
configuration as the baseline.

## Example B: a new feed-forward block (SwiGLU as the template)

`laboratory/blocks/feedforward.py` already contains the pattern for adding one:

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden=None, dropout=0.0):
        hidden = hidden or int(8 * d_model / 3)
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

Register it in `make_ffn`, add `ffn="swiglu"` to the experiment preset, and test
that parameter count and output shape match the paper's formula. Activation
functions are a one-line change; gating and routing are where the interesting
bugs live (shape mismatches only appear at the multiplication).

## Example C: swapping absolute positions for RoPE

This repository supports it as a configuration change, which is the point of
keeping positional information out of the model classes:

```python
model = GPT(..., pos="rope")        # head_dim must be even
```

The underlying implementation is in `laboratory/blocks/positional_encoding.py`
(`RotaryEmbedding`, `apply_rotary`), and `MultiHeadAttention` applies it after
splitting heads. To reproduce a paper that adds *length scaling* to RoPE, change
only `RotaryEmbedding.inv_freq` (for example, divide positions by a factor) and
keep the application code untouched. Then run the relative-position test and a
`smoke` run at both context lengths.

## Example D: a new recurrent cell

1. Add a cell class in `laboratory/models/recurrent.py`, with the gate order documented.
2. Add it to `RECURRENT_LAYERS` in the recurrent experiment (or a new
   experiment directory if the training regime differs).
3. Test it against a hand-written reference or against PyTorch's fused kernel
   with copied weights, exactly like the existing LSTM/GRU tests.
4. Add `tiny` and `smoke` presets and compare against LSTM at matched
   parameters on the same token budget.

The rule that keeps this repository useful: **every new cell or block must keep
the same forward signature as the one it replaces.** Then experiments stay
comparable and the smoke tests keep covering old paths.
