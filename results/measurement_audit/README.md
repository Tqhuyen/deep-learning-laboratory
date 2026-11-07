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
- One denoiser (DMID-d) supplies the protocol-shift table. A second and a third would turn it from an illustration into
  a budget.

## A second denoiser: Restormer, and a third protocol axis

The table above uses DMID-d's released outputs. To check whether the effect is a property of one model or of the
protocol, Restormer (σ-specific colour weights) is run on our own draws under the same scoring rules, plus one axis
DMID-d's outputs cannot test: a **clipped-and-quantised input**, the NTIRE convention.

Run: CPU kernel `protocol-effects-restormer-s25` / `-s50` (σ 15 data from `protocol-effects-restormer-small`, which
stopped after σ 15). Scores: [`restormer_protocols.csv`](restormer_protocols.csv),
[`restormer_protocol_shifts.csv`](restormer_protocol_shifts.csv).

**Shift against the unclipped full-image RGB score (dB), Restormer, σ 15:**

| set | GRL crop (RGB) | Y channel | clipped uint8 input |
|---|---:|---:|---:|
| CBSD68 | +0.011 | **+1.760** | −0.356 |
| Kodak24 | **−0.099** | **+1.822** | −0.320 |
| McMaster | +0.020 | **+2.811** | −0.776 |
| mean \|shift\| | 0.043 | **2.13** | 0.48 |

**Comparison with DMID-d (σ 15):**

| axis | DMID-d | Restormer | agreement |
|---|---|---|---|
| GRL crop, Kodak24 | −0.102 | −0.099 | **same to 0.003 dB** |
| GRL crop, CBSD68 | +0.004 | +0.011 | same sign, same size |
| GRL crop, McMaster | +0.013 | +0.020 | same sign, same size |
| Y channel, CBSD68 / Kodak24 / McMaster | +1.76 / +1.74 / +2.80 | +1.76 / +1.82 / +2.81 | **same to 0.08 dB** |
| clipped uint8 input | not measurable (outputs are already uint8) | −0.36 / −0.32 / −0.78 | new axis |

Three things follow.

1. **The protocol effect is a property of the protocol, not of the model.** A diffusion posterior mean and a
   convolutional transformer lose the same 0.10 dB to GRL's Kodak24 crop and the same 1.7–2.8 dB to the colour-space
   choice, to within 0.08 dB. This is what turns an illustration into a budget.
2. **The cropping effect is set-specific, not dataset-generic.** Kodak24 loses 0.10 dB; CBSD68 and McMaster gain
   0.01–0.02 dB. A leaderboard that crops uniformly therefore mis-ranks methods differently across sets.
3. **The input convention is a third axis and it is large.** Feeding the *same* method a clipped, quantised version of
   the same noisy image costs **0.32–0.78 dB**, which is 3–8× the recent SOTA margins. Combined with the Y-channel
   choice, a paper that does not state both conventions cannot be compared with one that does.
   This also bears on reported fine-tuning gains: the NTIRE 2026 report's "+3.37 dB over public Restormer σ 50" is
   measured against a model fed the NTIRE clipped input, and part of that gap is this convention, not the training.

**Caveat.** The clipping axis is measured at σ 15 only; σ 25 and σ 50 are running.
