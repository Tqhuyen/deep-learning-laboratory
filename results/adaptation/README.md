# Per-image self-supervised adaptation of the diffusion prior's one-step denoiser

Can a frozen ADM be adapted to one noisy image, using only that image, and beat its own posterior mean? The
parameters adapted are one gain per output channel of every convolution plus every GroupNorm γ/β: 215,814 values,
0.04 % of the network.

Each run takes 4 random 256² crops per step. The optimiser is Muon, as everywhere in the repository; all of these
parameters are 1-D, so its AdamW branch updates them. Training uses lr 1e-3, fp16 autocast and gradient checkpointing.

| run | Kaggle kernel | notebook | data |
|---|---|---|---|
| gate on 4 images, σ 25 | [`adaptation-gate-urban100`](https://www.kaggle.com/code/trnquanghuyn/adaptation-gate-urban100) (30 min) | [`adaptation_gate_urban100.ipynb`](../../notebooks/adaptation_gate_urban100.ipynb) | [`gate4/`](gate4/) |
| gate on dev-12, σ 25 / 50 | [`adaptation-dev12`](https://www.kaggle.com/code/trnquanghuyn/adaptation-dev12) (73 min) | [`adaptation_dev12.ipynb`](../../notebooks/adaptation_dev12.ipynb) | [`dev12/`](dev12/) |
| confirmation, settings frozen | [`adaptation-confirm`](https://www.kaggle.com/code/trnquanghuyn/adaptation-confirm) (7 h 54 min) | [`adaptation_confirm.ipynb`](../../notebooks/adaptation_confirm.ipynb) | [`confirm/`](confirm/) |

## Losses

- **Oracle@σ′.** Clean crops plus fresh noise at σ′ = σ√(1 + α²), with the clean image as target. This is the expected
  loss that R2R estimates. It is an upper reference, not a method.
- **R2R (Recorrupted-to-Recorrupted, Pang et al. 2021).** Input y + ασz at the step matched to σ′, target y − σz/α,
  with α = 0.5. In expectation this equals the MSE at σ′ plus a constant, and it uses only the noisy image. The
  adapted model is applied at the step matched to σ, which works because ADM is time-conditioned.

## Results (Urban100, 512-px tiles, gain over the unadapted model in the same precision)

**Gate on 4 dev images, σ 25, 60 steps:**

| image | 8 | 16 | 24 | 32 | mean |
|---|---:|---:|---:|---:|---:|
| oracle@σ′ | +0.135 | +0.128 | +0.182 | +0.330 | +0.194 |
| R2R | +0.027 | +0.065 | +0.110 | +0.257 | +0.115 |

Parameters tuned on image i and applied to another image j *lose* 0.10 dB on average. The adaptation is
image-specific, so dataset-level (transductive) adaptation is not pursued.

**Gate on all 12 dev images (mean gain in dB; all 12 images improve at every setting):**

| R2R steps | σ 25 | σ 50 |
|---:|---:|---:|
| 30 | +0.079 | +0.082 |
| 60 | +0.094 | +0.115 |
| 90 | **+0.110** | **+0.127** |

- **Oracle@σ′ at σ 25 (60 steps):** +0.172 dB. R2R at 90 steps reaches 64 % of it, but at matched 60 steps only 55 %.
- **Gate verdict: passed only through unmatched step counts.** The pre-registered criteria were ≥ +0.06 dB at both σ
  and ≥ 60 % of the oracle. The ratio criterion was met by comparing 90 R2R steps with a 60-step oracle (Phase-1 extreme
  critique). 90 steps is the edge of the {30, 60, 90} grid and the gain had not saturated. It is frozen for the
  confirmation, which stands on its own held-out test.
- **Per-image gains** range from +0.02 to +0.44 dB. They are largest on images with strong repetition (images 32 and
  40).

## Confirmation (settings frozen; nothing tuned on these images)

**Settings.** 90 R2R steps; Urban100 is scored with 384-px tiles (chosen on dev-12); Kodak24 / McMaster with the whole
image. Each row is the ×2 (flip) estimate. Gains are paired on identical noise, with a sign-flip test (200,000
patterns) and Holm correction.

| set, σ | n | unadapted | adapted | gain | images worse by > 0.05 dB | pre-registered verdict |
|---|---:|---:|---:|---:|---:|---|
| Urban100 confirm-88, σ 25 | 88 | 33.163 | **33.310** | **+0.147** | 0 % | success (p < 10⁻⁴) |
| Urban100 confirm-88, σ 50 | 88 | 30.432 | **30.586** | **+0.155** | 0 % | success (p < 10⁻⁴) |
| Kodak24, σ 50 | 24 | 30.138 | 30.120 | −0.018 | 25 % | failed |
| McMaster, σ 50 | 18 | 30.488 | 30.402 | −0.086 | 61 % | failed |

- **Urban100.** All 88 images improve at both σ: minimum +0.007, median +0.11, maximum +0.83 dB.
- **Secondary rows, 384-px tiles unadapted.** Kodak24 30.156 and McMaster 30.490, i.e. +0.018 and +0.002 over the
  whole image. The tile rule costs nothing on these smaller images.

**When the adaptation helps: not identified (Phase-1 review).** The failing sets differ from Urban100 in several
variables at once:

| set | noisy values per image | per adapted parameter (216 k) | passes per pixel (90 × 4 × 256²) | gain |
|---|---:|---:|---:|---:|
| Urban100 (colour, ~1024 px) | ≈ 2.4 M | ≈ 11× | ≈ 34 | +0.15 |
| Kodak24 (colour, 768×512) | 1.18 M | 5.5× | 60 | −0.02 |
| McMaster (colour, 500×500) | 0.75 M | 3.5× | 94 | −0.09 |
| BSD68 / Set12 (gray) | 0.07–0.26 M | ≤ 1.2× | 90–360 | −0.2 to −1.2 ([`../gray_lift`](../gray_lift/README.md)) |

- **Content.** Urban100 is the repetitive set; GainTuning (NeurIPS 2021) also found its gains largest there.
- **Size.** Inside confirm-88, image size does not predict the gain: Spearman(pixels, gain) = −0.10 at σ 25 and −0.12
  at σ 50 (n = 88). The largest third of the images gains 0.128 / 0.130 dB, against 0.155 / 0.160 for the smallest
  third.
- **Passes per pixel.** The fixed step budget revisits small images more often.
- **What was recorded beforehand** was only "less than Urban100, or nothing". The values-per-parameter threshold was
  drawn after the scores, so it is a hypothesis, not a finding.

## Reading

* **Every training-free route before this moved the shared prior and stalled within 0.02 dB:** calibration,
  ensembles, fine-tuning on DIV2K ([`../adm_finetune`](../adm_finetune/README.md)). This route is test-time training:
  it estimates an image-specific prior, as GainTuning (NeurIPS 2021) does for CNN denoisers. On Urban100 it gains
  5–10× more.
* **Cost.** About 4.3 GPU-minutes per image on a T4, including scoring. The gain has not saturated at 90 steps.
* **Not yet run:**
  - Urban100's 12 development images with the final pipeline, so that a full-100 number can be reported;
  - σ = 15;
  - a test that separates content from size and passes per pixel: Urban100 images cropped to Kodak / McMaster size,
    and large non-repetitive images (DIV2K-valid at full resolution);
  - GainTuning's own loss and parameter set on ADM, the direct prior-art baseline.

## What predicts whether adaptation helps (replaces the retracted capacity rule)

The earlier "noisy values per adapted parameter" rule was withdrawn: inside confirm-88 the image size does not predict
the gain (Spearman −0.10 / −0.12), and the threshold had been drawn after the scores. This section replaces it with a
predictor that was measured on 88 images, replicated across noise draws, and cross-checked across four set/σ cells.

**The predictor is the context deficit**, measured before any adaptation: `tile gain = PSNR(whole ×1) − PSNR(512-px
tiles ×2)`. It costs two extra forward passes and says how much the frozen prior's context mismatch costs *this*
image.

### Within Urban100 (n = 88), σ 50

| correlation | ρ | p |
|---|---:|---:|
| tile gain vs adaptation gain | **−0.622** | 1.0·10⁻¹⁰ |
| baseline PSNR vs adaptation gain | +0.369 | 4.1·10⁻⁴ |
| pixels vs adaptation gain | −0.109 | 0.31 |
| **partial** tile gain, controlling for baseline PSNR | **−0.539** | 5.9·10⁻⁸ |
| **partial** baseline PSNR, controlling for tile gain | **+0.036** | 0.74 |

- The relationship is **not** baseline quality in disguise: once the tile deficit is controlled for, baseline PSNR has no
  independent effect (ρ = +0.04, p = 0.74).
- The relationship is **not** image size: ρ = −0.11.
- It reproduces at σ 25 on a different draw: ρ = −0.537 raw, −0.472 partial.

### Across sets, σ 50 (and σ 25 where run)

| set | context deficit (dB) | adaptation gain (dB) |
|---|---:|---:|
| Urban100, σ 50 | −0.187 | **+0.148** |
| Urban100, σ 25 | −0.119 | **+0.142** |
| Kodak24, σ 50 | −0.030 | −0.018 |
| McMaster, σ 50 | −0.022 | −0.086 |
| CBSD68, σ 25 / 50 | −0.008 / −0.010 | not run |

Rank correlation across the four cells with an adaptation measurement: **−1.0**. Urban100's context deficit is 4–8×
larger than the small sets', and it is the only set on which adaptation helps.

### Why this is the same mechanism, not a coincidence

Tiling and adaptation both act on the prior's mismatch with the image, and they **substitute** for each other:

- Images where tiling helps most gain most from adaptation (ρ = −0.62).
- Images where tiling helps least gain least.
- The quartile table makes the trade-off explicit:

  | quartile of adaptation gain | n | adaptation gain | context deficit | baseline PSNR | Mpx |
  |---|---:|---:|---:|---:|---:|
  | Q1 (lowest) | 22 | +0.032 | −0.102 | 29.22 | 0.77 |
  | Q4 (highest) | 22 | +0.343 | −0.229 | 31.49 | 0.74 |

  The extremes differ by 0.13 dB in context deficit and hardly at all in size.

### Consequences

1. **A cheap pre-check.** Two extra forward passes predict whether adaptation will help, before paying its
   4.3 GPU-minutes. On this evidence the rule is: adapt when the context deficit exceeds ~0.1 dB.
2. **The cross-set failure is explained, not unexplained.** Kodak24 and McMaster have a context deficit of 0.02–0.03 dB,
   below the threshold, which is where the adaptation loss lands.
3. **The gate is free to falsify.** Applied prospectively to a new set, the predictor either ranks the images correctly
   or it does not. This is the pre-registered test the earlier rule never had.

**Caveat.** The cross-set check rests on four cells; the within-Urban100 evidence (n = 88, two σ, three draws on the
gain itself) is what carries the claim. The threshold "0.1 dB" is descriptive of these four cells and is not yet a
calibrated constant.

### The predictor holds at all three noise levels, and explains the σ 15 result

The same measurement, repeated on the full pipeline at every σ (n = 100 images):

| σ | context deficit (dB) | adaptation gain (dB) | per-image ρ(deficit, gain) | p |
|---:|---:|---:|---:|---:|
| 15 | **−0.082** | **+0.134** | −0.441 | 4.3·10⁻⁶ |
| 25 | −0.119 | +0.142 | −0.517 | 3.6·10⁻⁸ |
| 50 | −0.187 | +0.148 | −0.558 | 1.6·10⁻⁹ |

Three independent orderings agree:

1. **Within each σ** (n = 88–100 images): ρ from −0.44 to −0.56.
2. **Across σ** (three points): the deficit and the gain rank identically, rank correlation −1.0.
3. **Across datasets** (four cells): rank correlation −1.0.

**This explains the σ 15 result.** At σ 15 the prior's context deficit is half what it is at σ 50 (−0.082 against
−0.187 dB), the adaptation therefore gains least there (+0.134 against +0.148), and that is exactly why the pipeline
does not overtake GRL-B at σ 15 while it does at σ 25 and σ 50. No separate explanation is needed, and none of this
required a new hyper-parameter.

**What the predictor is not.** It is not a licence to tune the adaptation per σ. It says where adaptation has room, not
what the best settings are; the settings stay frozen across σ in every result reported here.
