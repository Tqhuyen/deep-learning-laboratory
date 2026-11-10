# Measurement audit: rescoring a published denoiser's released outputs

Published Gaussian-denoising SOTA margins are 0.00–0.11 dB. This study measures how much of a published PSNR depends
on the scoring protocol (test-set copy, cropping, orientation) while the denoiser and its outputs are held fixed.

- **Run.** CPU kernel [`trnquanghuyn/grl-outputs-audit`](https://www.kaggle.com/code/trnquanghuyn/grl-outputs-audit)
  (86 min) of [`notebooks/grl_outputs_audit.ipynb`](../../notebooks/grl_outputs_audit.ipynb).
- **Scores.** [`grl_rescored.csv`](grl_rescored.csv) per output image, [`grl_rescored_summary.csv`](grl_rescored_summary.csv)
  per set, model and σ.

## Method

- **Outputs.** GRL (CVPR 2023) released the denoised images of GRL-T / S / B (`visual_results_dn.tar.gz`, release
  v1.0.0). They cover colour CBSD68 / Kodak24 / McMaster / Urban100 and gray Set12 / BSD68 / Urban100, at σ 15 / 25 / 50.
- **Clean copies.**
  - Colour: DMID's release.
  - Gray: Set12 / BSD68 from the DnCNN repository.
  - Gray Urban100: the rounded BT.601 luma of DMID's colour copy.
- **Matching.** Each output is matched by content to a clean image and a crop offset: top-left, centre, bottom-right
  and the 1-px border crop, plain or transposed.
- **Scoring.** PSNR on the overlap, RGB or gray, on uint8 values.

## Results (GRL-B, released outputs, rescored)

| set | σ 15 / 25 / 50 published | rescored here | difference | output size vs clean |
|---|---|---|---|---|
| gray Set12 | 33.47 / 31.12 / 28.03 | 33.461 / 31.121 / 28.017 | −0.009 / +0.001 / −0.013 | full |
| gray BSD68 | 32.00 / 29.54 / 26.60 | 32.001 / 29.550 / 26.625 | +0.001 / +0.010 / +0.025 | **320×480 of 321×481** |
| gray Urban100 | 34.09 / 31.80 / 28.59 | 34.060 / 31.776 / 28.582 | −0.030 / −0.024 / −0.008 | full; our gray copy is a luma conversion |
| colour Kodak24 | 35.43 / 33.02 / 29.93 | 35.446 / 33.026 / 29.948 | +0.016 / +0.006 / +0.018 | **central 500×500 crop cut to 496×496 (62.6 % of the pixels)** |
| colour Urban100 | 35.54 / 33.35 / 30.46 | 35.532 / 33.337 / 30.450 | −0.008 / −0.013 / −0.010 | full |
| colour CBSD68, 67 images | 34.45 / 31.82 / 28.62 (68 images) | 34.486 / 31.858 / 28.660 | (+0.04, not comparable: n = 67) | 320×480 of 321×481 |

## What this shows

1. **Published numbers reproduce from released outputs.** Every set agrees with GRL's table within 0.03 dB:
   - gray Set12 / BSD68 / Urban100;
   - colour McMaster / Urban100 / Kodak24, on GRL's crop.
2. **Each method's table uses its own cropping.** GRL's BSD68 / CBSD68 outputs drop the last row and column; its
   McMaster outputs are the top-left 496×496. DMID's numbers and ours are computed on full images, so a leaderboard
   that mixes them compares different pixel sets.
3. **GRL's Kodak24 is not the standard Kodak24.** Its outputs are the central 500×500 crop of each image, cut to
   496×496:
   - offset (6, 134) in landscape, (134, 6) in portrait;
   - 62.6 % of the pixels;
   - on that crop they reproduce the published 35.43 / 33.02 / 29.93 within 0.02 dB
     ([`grl_kodak_cbsd68_located.csv`](grl_kodak_cbsd68_located.csv);
     [`notebooks/grl_outputs_audit_kodak.ipynb`](../../notebooks/grl_outputs_audit_kodak.ipynb)).

   The numbers are therefore not comparable with full-image Kodak24 results such as DMID 35.51 / 33.12 / 30.14 or
   Restormer 35.47 / 33.04 / 30.01.
4. **Test-set copies differ.** DMID's CBSD68 copy stores image 148026 rotated by 180° relative to the copy GRL used;
   after rotation it matches.

## The size of the protocol effect, with the denoiser held fixed

Run: CPU kernel [`trnquanghuyn/denoise-protocol-effects`](https://www.kaggle.com/code/trnquanghuyn/denoise-protocol-effects)
(5.5 min) of [`notebooks/denoise_protocol_effects.ipynb`](../../notebooks/denoise_protocol_effects.ipynb). Scores:
[`dmid_protocols.csv`](dmid_protocols.csv), [`dmid_vs_grl.csv`](dmid_vs_grl.csv),
[`dmid_vs_grl_summary.csv`](dmid_vs_grl_summary.csv).

DMID-d's released full-image outputs are scored under each protocol, so nothing but the scoring rule changes.

**Shift against the full-image RGB score (dB, σ 25 / σ 50):**

| protocol | CBSD68 | Kodak24 | McMaster | Urban100 |
|---|---:|---:|---:|---:|
| GRL's crop (top-left to a multiple of 8; Kodak central 500 cut to 496) | +0.005 / +0.006 | **−0.093 / −0.116** | +0.017 / +0.019 | 0.000 / 0.000 |
| shave 8 px from every border | −0.050 / −0.047 | −0.053 / −0.050 | +0.032 / +0.041 | −0.017 / −0.010 |
| **Y channel instead of RGB** | **+1.65 / +1.57** | **+1.64 / +1.52** | **+2.56 / +2.35** | **+2.19 / +1.98** |

- Cropping to a multiple of 8 moves a fixed method by up to 0.12 dB; shaving 8 px moves it by up to 0.05 dB.
- Scoring on the Y channel instead of RGB moves it by **1.5–2.6 dB**. A paper that does not say which it reports is not
  comparable at all.

## Does the published ordering survive a common protocol?
## Which protocol axes reorder methods, and which only shift the level

Two questions hide behind "the protocol moves numbers":

1. Does an axis change a method's *level*? All three do, and the previous sections quantify how much.
2. Does it change the *ordering* between two methods? Only some do, and it is worth separating them because a reviewer
   will ask whether the whole leaderboard is meaningless or merely imprecise.

DMID-d and Restormer, both measured under each axis, over the nine set–σ cells:

| axis | DMID-d − Restormer, full-image RGB | under the axis | sign changes | largest change in the gap |
|---|---:|---:|---:|---:|
| GRL's crop | +0.033 … +0.222 dB | +0.031 … +0.211 dB | **0 / 9** | 0.011 dB |
| Y channel | +0.033 … +0.222 dB | +0.033 … +0.206 dB | **0 / 9** | 0.023 dB |

**So scoring conventions mostly shift the level, not the ranking.** When a gap is large relative to the axis's shift —
here 0.03–0.22 dB against a 0.01–0.02 dB effect — the ordering survives. That is a weaker and more accurate statement
than "leaderboards are meaningless", and it is the right one.

**What did reorder methods was not a scoring convention.** The Kodak24 reversal in the section above came from GRL
evaluating on the **central 500×500 crop**, 62.6 % of each image — a different set of pixels, not a trimmed border. The
distinction matters:

- a *convention* (which border, which colour space, which input quantisation) moves levels by 0.01–2.5 dB and largely
  preserves orderings;
- a *different evaluation region* changes the underlying data and can reverse them, as it did for DMID-d against GRL-B
  at σ 15 and σ 25.

**Consequence for this paper.** The claim in C1/C2 is stated on the same pixels for both methods, which is the
condition under which orderings are reliable. The claim in C10 is about the size of the level shifts, which is where
the damage is: a 3.8 dB convention difference can promote a method across a leaderboard, and 0.05 dB margins are inside
the convention noise even when the ordering happens to be right.


DMID-d and GRL-B are scored on exactly the same pixels (GRL's crop region; each on its own noise draws):

| set, σ | published gap (DMID − GRL) | gap on the same pixels | sign |
|---|---:|---:|---|
| CBSD68 σ 15 | 0.00 | −0.001 | **flipped** |
| Kodak24 σ 15 | **+0.08** | **−0.041** | **flipped** |
| Kodak24 σ 25 | +0.10 | −0.001 | **flipped** |
| Kodak24 σ 50 | +0.21 | +0.075 | same |
| McMaster σ 50 | +0.14 | +0.160 | same |
| Urban100 σ 15 | −0.28 | −0.271 | same |

**Reading.**

- On Kodak24 at σ 15 and σ 25 the published ordering reverses once both methods are scored on the same pixels: DMID-d is
  0.08–0.10 dB ahead on the tables and 0.04 dB behind on the pixels.
- Urban100 shows the opposite: GRL-B is ahead by 0.17–0.27 dB on both accountings, so that gap is real.
- The protocol effect is therefore not uniform across sets. That is what makes a leaderboard that mixes protocols
  uninterpretable at the 0.1 dB scale.

## Caveats

- Each method is on its own noise draw, so this measures the protocol effect only up to draw noise. The draw term was
  measured separately: re-deriving the noise changes per-image PSNR by about 0.001 dB, so set means move by about
  0.001 dB.
- Two denoisers now supply the shifts (DMID-d and Restormer, next section), and they agree to 0.02 dB. A third, and a
  second independent implementation of the scoring code, would make it a full budget.

## A second denoiser: Restormer, and a third protocol axis

The tables above use DMID-d's released outputs. To check whether the effect is a property of one model or of the
protocol, Restormer (σ-specific colour weights) is run on our own draws under the same scoring rules, plus one axis
DMID-d's outputs cannot test: a **clipped-and-quantised input**, the NTIRE convention.

Runs: CPU kernels `protocol-effects-restormer-s25` and `-s50`; σ 15 (and Urban100, and the first three sets at σ 25)
from `protocol-effects-restormer` and `protocol-effects-restormer-small`. The full-grid kernel was cancelled at the
12-hour limit after finishing Urban100 σ 15 and the first three sets at σ 25; Urban100 at σ 25 and σ 50 was not run. Scores: [`restormer_protocols.csv`](restormer_protocols.csv),
[`restormer_protocol_shifts.csv`](restormer_protocol_shifts.csv).

**Shift against the unclipped full-image RGB score (dB), Restormer:**

| set | σ | GRL crop (RGB) | Y channel | clipped uint8 input |
|---|---:|---:|---:|---:|
| CBSD68 | 15 | +0.011 | +1.760 | −0.356 |
| CBSD68 | 25 | +0.010 | +1.658 | −0.690 |
| CBSD68 | 50 | +0.010 | +1.580 | **−1.742** |
| Kodak24 | 15 | **−0.099** | +1.822 | −0.320 |
| Kodak24 | 25 | **−0.086** | +1.709 | −0.617 |
| Kodak24 | 50 | **−0.106** | +1.621 | **−1.833** |
| McMaster | 15 | +0.020 | +2.811 | −0.776 |
| McMaster | 25 | +0.025 | +2.559 | **−1.658** |
| McMaster | 50 | +0.031 | +2.343 | **−3.807** |
| Urban100 | 15 | −0.000 | +2.432 | −0.642 |

Urban100 is the set the paper's headline claim lives on, so its row matters twice over.

- **The copy is validated.** Restormer scores **35.133** on the full 100 Urban100 images at σ 15, against a published
  35.13. That is the direct check for a copy-level defect on the set where C1 and C2 are made.
- **No convention mismatch in the headline comparison.** GRL does not clip and neither do we, so the +0.065 / +0.220 dB
  margins are between two methods evaluated under the same input convention. Had either side clipped, the gap would
  carry a 0.6–2.5 dB term instead.
- **The crop axis is empty on Urban100**, as it is for DMID-d, because GRL's Urban100 outputs cover the full image.

**Agreement with DMID-d, measured independently on its own released outputs, over all nine set–σ cells:**

| axis | largest disagreement between the two models |
|---|---|
| GRL crop | **0.012 dB** |
| Y channel | **0.024 dB** |

Four conclusions.

1. **The protocol effect belongs to the protocol, not to the model.** A diffusion posterior mean and a convolutional
   transformer lose the same amount to the same scoring choices, over nine cells, to within 0.02 dB. That is what turns
   an illustration into a budget.
2. **The cropping effect is set-specific, not dataset-generic.** Kodak24 loses 0.09–0.11 dB; CBSD68 and McMaster gain
   0.01–0.03 dB. A leaderboard that crops uniformly therefore mis-ranks methods differently across sets, which is why
   the published Kodak24 ordering of two methods reverses on identical pixels (see above).
3. **The colour-space choice is large and shrinks with σ:** 1.58–2.81 dB at σ 15, 1.62–2.56 dB at σ 25, 1.58–2.34 dB at
   σ 50.
4. **The input convention is the largest axis, and it grows steeply with σ.** Mean over the three sets:
   **−0.48 dB at σ 15, −0.99 dB at σ 25, −2.46 dB at σ 50**; McMaster alone reaches **−3.81 dB** at σ 50. Neither
   model's published protocol clips, and NTIRE's does. A 3.8 dB term is 40× a recent SOTA margin, so any comparison
   between a clipped-protocol number and an unclipped one is meaningless, and it also means part of the NTIRE 2026
   report's "+3.37 dB over public Restormer σ 50" is a convention rather than a training effect.

**What the clipping number does and does not say.**

- **It is not a surprise that clipping hurts.** Clipping at 0 and 255 throws information away, so every method, including
  an oracle, must do worse on a clipped input. The larger σ is, the larger the fraction of pixels pinned to the
  boundary, and the effect is monotone in σ on all three sets, which is what the mechanism predicts.
- **What is worth reporting is the size.** 0.48 / 0.99 / 2.46 dB at σ = 15 / 25 / 50, and 3.81 dB on McMaster, is
  5–40× a recent SOTA margin. A clipped-protocol number and an unclipped one therefore describe different problems, and
  a leaderboard that mixes them is not ranking methods.
- **It is routinely unstated.** Of the papers whose protocols this project read, neither DMID nor Restormer clips, NTIRE
  clips and quantises, and most reports do not say which convention produced their numbers.

**Caveat.** The clipping arm feeds Restormer the rounded, clipped image, so it carries a small quantisation cost the
float arm does not. That is not a confound but the quantity of interest: it is exactly how the NTIRE protocol differs
from the KAIR one.
