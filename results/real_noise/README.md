# Zero-shot real-noise denoising with pretrained diffusion priors

Three Kaggle runs (private, 2× Tesla T4):

* [`trnquanghuyn/real-noise-zero-shot`](https://www.kaggle.com/code/trnquanghuyn/real-noise-zero-shot) (48 min) runs
  [`notebooks/real_noise_zero_shot.ipynb`](../../notebooks/real_noise_zero_shot.ipynb);
* [`trnquanghuyn/real-noise-round2`](https://www.kaggle.com/code/trnquanghuyn/real-noise-round2) (44 min) runs
  [`notebooks/real_noise_round2.ipynb`](../../notebooks/real_noise_round2.ipynb);
* [`trnquanghuyn/real-noise-round3`](https://www.kaggle.com/code/trnquanghuyn/real-noise-round3) (114 min) runs
  [`notebooks/real_noise_round3.ipynb`](../../notebooks/real_noise_round3.ipynb).

The test sets are CC (15), PolyU (100) and FMDD (48), all 512×512, from DMID's release. PSNR and SSIM are computed on
RGB uint8 (FMDD: the channel mean, as DMID does).

**Bottom line.** Nothing here beats DMID-d (TPAMI 2024). Its accuracy is reached with 21–31× fewer network
evaluations in the second stage.

* **Pixel-shuffle downsampling plus the one-step ADM posterior mean** reaches CC 36.3, PolyU 37.9–38.0 and FMDD
  30.9 dB at its best settings. DMID-d's released outputs score 37.99, 38.62 and 33.40 dB.
* **DMID's per-image noise transformation does most of the work.** On its released transformed images:
  * one step reproduces DMID-p exactly;
  * DMID's own stochastic three-step chain, averaged over **16 samples with half of them run on the flipped image
    (48 evaluations)**, ties DMID-d's 500 or 333 samples (1,500 or 999 evaluations): +0.004, +0.000, −0.001 dB;
  * extrapolating the sample average to infinitely many samples adds about 0.01 dB, so this stage has no accuracy
    left to win.

## Checks

* **DMID's released outputs, rescored with our code,** reproduce its Table 3: CC 37.99, PolyU 38.62, FMDD 33.40
  (DMID-d); 37.09, 38.46, 33.09 (DMID-p). Every clean image matches ours exactly.
* **Restormer trained on SIDD (supervised)** scores 36.31 / 37.63 / 31.81 here, against 36.27 / 37.51 / 31.81 in
  DMID's table.
* **The transformed images map onto the noisy inputs.**
  * CC and PolyU match one-to-one by MSE, in DMID's index order.
  * FMDD repeats each ground truth over 4 noise realisations, so its matches are not unique; it uses DMID's index
    order, which agrees with the matches on 45 of 48 images.
  * DMID's chain reproduced with our code reaches 33.40 dB on FMDD, as released.

## Camera noise is strongly correlated

Means over images ([`round1/noise_stats.csv`](round1/noise_stats.csv)):

| | residual std | lag-1 corr. | lag-2 corr. | σ̂ full | σ̂ PD2 | σ̂ PD4 |
|---|---:|---:|---:|---:|---:|---:|
| CC | 0.0216 | 0.72 | 0.39 | 0.0053 | 0.0137 | 0.0225 |
| PolyU | 0.0161 | 0.68 | 0.40 | 0.0021 | 0.0065 | 0.0109 |
| FMDD | 0.0463 | 0.22 | 0.08 | 0.0272 | 0.0351 | 0.0390 |

* **The wavelet estimate misses most of the noise on the full image:** 4–10× low on CC and PolyU.
* **It needs PD4 (stride 4)** to match the residual std on CC. Correlation is still 0.4 at lag 2.

## Results

Mean PSNR in dB over all images. Scores are in [`round1/all_scores.csv`](round1/all_scores.csv),
[`round2/all_scores.csv`](round2/all_scores.csv) and [`round3/all_scores.csv`](round3/all_scores.csv).

**Without DMID's noise transformation (rounds 1–2):**

| method | evaluations | CC | PolyU | FMDD |
|---|---:|---:|---:|---:|
| noisy input | 0 | 33.41 | 35.97 | 27.22 |
| ADM, full image, step from σ̂(PD4) × 2 | 1 | 36.00 | 37.15 | 30.74 |
| ADM after PD4, m = 1.25 | 1 | 36.29 | 37.32 | 29.03 |
| ADM after PD4, m = 1.75 (chosen on PolyU 1–50) | 1 | 35.34 | 37.70 | 30.13 |
| ADM after PD4 (m = 2), then a full-resolution step at 0.5 σ̂ | 2 | 35.00 | 37.88 | 30.86 |
| DRUNet after PD2 (Gaussian, given σ̂) | 1 | 34.88 | 36.46 | 28.52 |
| Restormer, SIDD-trained (supervised) | 1 | 36.31 | 37.63 | 31.81 |

**On DMID's noise-transformed images (round 3):**

| method | evaluations | CC | PolyU | FMDD |
|---|---:|---:|---:|---:|
| one step, m = 0.7 | 1 | 37.25 | 38.46 | 32.00 |
| DMID-p, released | 1 (CC, PolyU) / 3 (FMDD) | 37.09 | 38.46 | 33.09 |
| deterministic chain (η = 0), DMID's steps | 3 | 37.86 | 38.59 | 33.16 |
| deterministic chain + flip | 6 | 37.88 | 38.60 | 33.17 |
| stochastic chain (η = 0.85), 1 sample | 3 | 37.85 | 38.56 | 33.22 |
| stochastic chain, 8 samples | 24 | 37.97 | 38.61 | 33.38 |
| stochastic chain, 8 samples, half flipped | 24 | 37.98 | 38.62 | 33.39 |
| **stochastic chain, 16 samples, half flipped** | **48** | **37.99** | **38.62** | **33.40** |
| **DMID-d, released** | **1,500 / 999 / 999** | **37.99** | **38.62** | **33.40** |

Paired differences in PSNR (dB) against DMID-d's released outputs. Each cell gives the mean, its 95 % interval and
the number of images where ours is higher:

| method | CC | PolyU | FMDD |
|---|---|---|---|
| 16 samples, half flipped | +0.004 ± 0.008 (7/15) | −0.001 ± 0.002 (41/100) | +0.000 ± 0.004 (25/48) |
| 8 samples, half flipped | −0.004 ± 0.007 (4/15) | −0.005 ± 0.002 (24/100) | −0.013 ± 0.007 (11/48) |
| 8 samples | −0.018 ± 0.004 (0/15) | −0.009 ± 0.002 (8/100) | −0.022 ± 0.006 (2/48) |
| deterministic chain + flip | −0.105 ± 0.023 (0/15) | −0.022 ± 0.007 (15/100) | −0.231 ± 0.087 (11/48) |

**Flipping half the samples** adds +0.014 (CC, 13/15 images), +0.004 (PolyU, 79/100) and +0.009 dB (FMDD, 36/48) at
an equal count of 8 samples.

**Cost.** On a T4, 16 samples take 46–51 s per 512×512 image, so DMID-d's 500 samples take about 25 min.

## Reading

* **PD trades two errors against each other.** It decorrelates the noise only at stride 4, where each 128×128 sub-image
  is an aliased quarter-resolution view. The prior then has little to work with, and merging the sub-images leaves
  seams. No step multiplier, stride, flip, refinement or blend got within 1.6 dB of DMID-d on CC.
* **DMID's per-image noise transformation does most of the work.** Even its one-step denoise (DMID-p) beats every PD
  variant.
* **On transformed images, the multi-step chain matters and the sample count barely does.**
  * Unlike on Gaussian noise, one posterior-mean step is 0.2–1.4 dB short of the chain. The transformed noise is not
    the Gaussian of the stated level, and re-noising between steps absorbs the mismatch.
  * Averaging is cheap to saturate: the gain from 8 to 500 samples is 0.02 dB.
  * Flip-balanced sampling closes half of that gap at no cost.
* **Accuracy beyond DMID-d has to come from the transformation**, not from the denoising chain. Not tried: a new
  noise transformation, and signal-dependent (variance-stabilising) models for FMDD.

## Two regimes on the transformed images (analysis of round 3; no new run)

The comparison is per image between two estimators on DMID's transformed images:
- the best one-step ×2 (m ∈ {0.4, 0.55, 0.7});
- the 16-sample chain.

| | n | one step wins | per-image oracle − chain (dB) | corr(lag-1 noise correlation, chain − one step) |
|---|---:|---:|---:|---:|
| CC | 15 | 4 | +0.084 | 0.51 |
| FMDD | 48 | 13 | +0.084 | 0.63 |
| PolyU | 100 | 14 | +0.021 | −0.00 |

**Reading (corrected after the Phase-1 review).**
- **The pooled correlations are between-group effects.** Within a camera or modality they vanish or change sign.
  - CC: Canon 5D Mark III r = +0.25 (n = 3), Nikon D600 −0.21 (n = 3), Nikon D800 −0.05 (n = 9). Pooled Spearman is
    0.41 (p = 0.13).
  - FMDD: Confocal BPAE −0.31, TwoPhoton BPAE −0.14, WideField BPAE +0.27 (n = 12 each). The pooled 0.63 is mainly
    WideField (lag-1 0.58–0.65, chain + 2.85 dB) against the rest (lag-1 ≤ 0.24).
  - The lag-1 correlation therefore identifies the camera or modality, not a per-image mechanism.
- **The one-step comparator is under-tuned.** Its best m sits on the edge of the grid {0.4, 0.55, 0.7} for 14 of 15
  CC images, 42 of 48 FMDD images and 98 of 100 PolyU images. The chain's advantage includes the cost of that grid.
- A per-image switch is worth at most 0.08 dB, and only if it could be chosen blind. That would not clear the newer
  bars: Cheng & Tan (NeurIPS 2024) report CC 38.01 / PolyU 38.71, and P2N (CVPR 2025) FMDD 34.00.
