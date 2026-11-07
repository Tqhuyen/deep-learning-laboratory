# Is the Urban100 tile gain a hidden ensemble?

The tile pipeline blends overlapping 512-px tiles with linear ramps, so pixels inside an overlap average two to four
tile estimates per pass; the ×2 flip runs the tiling on the mirrored image, so its grid is anchored at the other edge.
That raises an obvious objection: part of the "tiling helps" gain could just be ensembling.

- **Run.** CPU kernel [`urban100-hidden-ensemble`](https://www.kaggle.com/code/trnquanghuyn/urban100-hidden-ensemble)
  (4 min) of [`notebooks/urban100_hidden_ensemble.ipynb`](../../../notebooks/urban100_hidden_ensemble.ipynb), on the saved
  float16 pilots from `urban100-tiles`.
- **Scores.** [`regions.csv`](regions.csv) per image and region; [`summary.json`](summary.json) the test.

## Design

On pixels covered by **exactly one** tile per pass — 41.9 % of each image — tiles ×2 and whole ×2 each average exactly
two estimates. Any difference there is the restricted context, not averaging. The test is pre-registered: the context
account survives if the single-coverage gain is at least half the all-pixel gain and is positive with sign-flip
p < 0.01 at σ 25 and σ 50.

Consistency check: the all-pixel gain recomputed from the float16 pilots matches the GPU run's per-image score table to
within **0.0009 dB**.

## Result

| σ | gain, all pixels | gain, single-coverage | gain, the rest | share | single-coverage images better |
|---:|---:|---:|---:|---:|---:|
| 15 | +0.053 | **+0.044** | +0.058 | 84 % | 83/88 |
| 25 | +0.087 | **+0.076** | +0.094 | 87 % | 88/88 |
| 50 | +0.147 | **+0.130** | +0.158 | 88 % | 88/88 |

**Verdict: the context account survives.** Four-fifths to seven-eighths of the gain is still there on pixels with no
overlap averaging, at every σ, on 88 of 88 images at σ 25 and σ 50, with p < 10⁻⁵.

Overlap averaging contributes at most 0.009–0.028 dB, and only on the overlapped pixels.

## What this does not establish

The test separates *averaging* from *context restriction*. It does not say **which part** of the context restriction
matters: attention token count, the locality of GroupNorm statistics, or the convolutional receptive field. Attention
rescaling alone recovers about half the gain, which bounds the attention share from below but not from above. That arm
was not run.

## A by-product worth recording

The ×2 flip gains more on tiles (+0.034 / +0.041 / +0.053 dB) than on whole images (+0.026 / +0.028 / +0.033 dB). Part
of that is a grid-shift ensemble, because the mirrored pass anchors the tiling at the other edge. The clean statement of
the tile gain is therefore the **×1** contrast, `tiles ×1 − whole ×1` = +0.044 / +0.074 / +0.127 dB, which does not
involve the flip at all.
