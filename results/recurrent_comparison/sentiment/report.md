# Sentiment classification: RNN vs GRU vs LSTM

Tier: `development`. Seeds: [0, 1, 2].

Parameter matching: per-layer cell parameters are
`m * (I*H + H^2 + 2H)` with `m = 1` (RNN), `3` (GRU), `4` (LSTM);
the hidden size is solved for the shared budget.

| cell | hidden | parameters | accuracy (mean ± std) | NLL | gradient slope |
|---|---:|---:|---:|---:|---:|
| rnn | 1158 | 2,000,190 | 0.6743 ± 0.0105 | 0.7486 | -0.13333 |
| gru | 655 | 1,998,557 | 0.7393 ± 0.0037 | 0.6985 | -0.12982 |
| lstm | 563 | 1,998,116 | 0.7393 ± 0.0134 | 0.7162 | -0.12789 |

| cell | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| rnn | 0.6720 | 0.6651 | 0.6858 |
| gru | 0.7408 | 0.7351 | 0.7420 |
| lstm | 0.7248 | 0.7420 | 0.7511 |

| pair | accuracy difference | 95% paired-bootstrap CI | McNemar p | Holm p |
|---|---:|---|---:|---:|
| rnn - gru | -0.0650 | [-0.0872, -0.0420] | 0.0002389 | 0.0007168 |
| rnn - lstm | -0.0650 | [-0.0856, -0.0443] | 0.002339 | 0.004677 |
| gru - lstm | +0.0000 | [-0.0153, +0.0149] | 0.2467 | 0.2467 |

McNemar uses seed-0 predictions; the bootstrap interval uses seed-averaged
per-example correctness. Neither accounts for seed variability, so the
per-seed table above is the stricter check. A CI that excludes zero and a
Holm-corrected McNemar p < 0.05 support a difference on this test set;
neither establishes a universal ordering.
