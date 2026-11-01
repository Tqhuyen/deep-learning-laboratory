# Recurrent language modelling: RNN vs GRU vs LSTM

Tier: `development`. Seeds: [0, 1, 2].
Training tokens per run: 8,192,000.

Parameter matching: per-layer cell parameters are
`m * (I*H + H^2 + 2H)` with `m = 1` (RNN), `3` (GRU), `4` (LSTM);
hidden size is solved for the shared budget. BPC = CE / ln 2.

| cell | hidden | parameters | validation BPC (mean ± std) | gradient slope |
|---|---:|---:|---:|---:|
| rnn | 1159 | 1,499,335 | 2.3842 ± 0.0075 | -0.01148 |
| gru | 664 | 1,501,545 | 2.4065 ± 0.0055 | -0.01138 |
| lstm | 572 | 1,501,149 | 2.5463 ± 0.0149 | -0.01128 |

| cell | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| rnn | 2.3883 | 2.3888 | 2.3755 |
| gru | 2.4117 | 2.4008 | 2.4071 |
| lstm | 2.5348 | 2.5411 | 2.5631 |

| pair | Δ BPC | 95% paired-bootstrap CI | bootstrap p | Holm p |
|---|---:|---|---:|---:|
| rnn - gru | -0.0223 | [-0.0254, -0.0192] | 0 | 0 |
| rnn - lstm | -0.1622 | [-0.1674, -0.1570] | 0 | 0 |
| gru - lstm | -0.1398 | [-0.1434, -0.1361] | 0 | 0 |

A negative Δ BPC means the first cell has lower bits per character.
The bootstrap resamples validation windows, so its interval can be much
narrower than run-to-run spread; the per-seed table is the stricter check.
CIs quantify uncertainty on this corpus, not a universal ordering.

Samples are in `samples.txt` (same prompt, same sampling seed per cell).
