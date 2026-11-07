# Grayscale Gaussian denoising with an RGB diffusion prior: the isotropic chroma-noise lift

Kaggle run [`trnquanghuyn/gray-lift-kill-test`](https://www.kaggle.com/code/trnquanghuyn/gray-lift-kill-test)
(private, 2× Tesla T4, 34 min) of [`notebooks/gray_lift_kill_test.ipynb`](../../notebooks/gray_lift_kill_test.ipynb).
Scores: [`scores.csv`](scores.csv) (per image, σ, seed and method) and [`summary.json`](summary.json).

**Bottom line.**
- **Training-free result.** One evaluation of the frozen ADM-256 ImageNet prior, with the flip average, reaches
  32.04 / 29.62 / 26.75 dB on BSD68 at the published seed. No diffusion denoiser had reported gray Set12 or BSD68
  results before.
- **On identical draws** it beats the released Restormer, Xformer, SCUNet and DRUNet gray models; the GRL-B rerun is
  pending.
- **Against published numbers** the margins are +0.01–0.04 dB at BSD68 σ 15 and Set12 σ 15 / 25. They are unpaired,
  so they are not claimed as wins.
- **Not the overall state of the art.** PTG-RM (CVPR 2024), a supervised backbone refined with CLIP features, reports
  32.29 / 29.81 / 26.91 dB. That is 0.14–0.26 dB higher in every Set12 and BSD68 cell, but no denoising weights are
  released.

## Method

A gray observation $y = x + \sigma n$ is lifted to RGB as $Y = y\mathbf 1 + w$, with
$w \sim \mathcal N\big(0,\, 3\sigma^2 (I - uu^\top)\big)$ and $u = \mathbf 1/\sqrt 3$.

- The noise in $Y$ is then exactly isotropic: $\sigma^2\mathbf 1\mathbf 1^\top + 3\sigma^2(I - uu^\top) = 3\sigma^2 I$.
  The luminance, the sufficient statistic, is unchanged.
- The one-step ADM posterior mean is taken at the step matched to $\sqrt 3\,\sigma$, then averaged over channels.
- **Estimators:** ×1 is one chroma draw. ×2 adds its horizontal flip. ×4 (the pre-registered estimator) uses two chroma
  draws, each with its flip.
- **Naive alternative, for comparison:** DMID's handling of gray images replicates y to three channels without chroma
  noise and embeds it at the step for σ.

## Protocol

- **Noise:** Restormer's `test_gaussian_gray_denoising.py`. The float32 image is divided by 255, `np.random.seed(seed)`
  is set per image, then `img += np.random.normal(0, σ/255, shape)`. There is no clipping, and outputs are rounded to
  uint8.
- **Seeds:** seed 0 is the published protocol; seeds 1 and 2 replicate it.
- **Images:** BSD68 and Set12 from github.com/cszn/DnCNN.
- **Validity check:** Restormer-gray (σ-specific weights) is rerun on the same draws. It lands within 0.005 dB of its
  published numbers.
- **Identical draws across jobs.** The lift job drew its noise from NumPy's global RNG inside two GPU threads. A CPU
  rerun of Restormer-gray on single-threaded seed-0 draws reproduces the job's per-image PSNR on all 240 (image, σ)
  pairs, with a largest difference of 0.00006 dB. So the seed-0 pairings with the single-threaded reruns (Xformer,
  SCUNet, DRUNet, GRL-B) are on identical draws ([`draw_check/`](draw_check/),
  [`notebooks/gray_draw_check.ipynb`](../../notebooks/gray_draw_check.ipynb)).
- **Pre-registration:** the success and kill criteria were fixed before the run, against the bars then known:
  Xformer / GRL-B. PTG-RM was found afterwards.

## Results

Mean PSNR in dB, seed 0 (the published protocol):

| method | evaluations | BSD68 σ 15 | 25 | 50 | Set12 σ 15 | 25 | 50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ADM, replicated to RGB (DMID-style) | 1 | 27.02 | 22.89 | 17.45 | 27.16 | 22.90 | 17.36 |
| ADM lift ×1 | 1 | 32.028 | 29.607 | 26.728 | 33.489 | 31.193 | 28.208 |
| ADM lift ×2 (flip) | 2 | 32.039 | 29.623 | 26.753 | 33.501 | 31.211 | 28.230 |
| ADM lift ×4 (2 draws × flip) | 4 | **32.040** | **29.624** | **26.754** | 33.503 | 31.212 | 28.232 |
| Restormer-gray, our rerun | 1 | 31.965 | 29.524 | 26.617 | 33.416 | 31.077 | 28.000 |
| best published, no external prior | — | 32.00 GRL-B | 29.55 Xformer | 26.67 SCUNet | 33.49 HAT | 31.19 Hi-IR | 28.15 Hi-IR |
| PTG-RM (supervised + CLIP), best variant | — | 32.29 | 29.81 | 26.91 | 33.74 | 31.30 | 28.37 |

The DMID-style row is the mean over seeds 0–2; the other ADM and Restormer rows are seed 0.

- **Against the best published without an external prior:** BSD68 +0.04 / +0.07 / +0.08 dB; Set12 +0.01 / +0.02 /
  +0.08 dB.
- **Against PTG-RM:** −0.25 / −0.19 / −0.16 dB on BSD68 and −0.24 / −0.09 / −0.14 dB on Set12.

Paired differences over seeds 0–2, ×4 − Restormer-gray (dB; images better / pairs):

| | σ 15 | σ 25 | σ 50 |
|---|---:|---:|---:|
| BSD68 | +0.075 (204/204) | +0.100 (204/204) | +0.138 (204/204) |
| Set12 | +0.087 (35/36) | +0.139 (36/36) | +0.230 (36/36) |

## Reading

- **The lift is what makes the RGB prior usable on gray images.** Replicating y into three channels gives the network
  perfectly correlated "per-channel" noise that it was never trained on. The result is 5–9 dB worse than the lift and
  only 2.4–3.3 dB better than the noisy input.
- **The flip is worth +0.011 to +0.021 dB, as in colour. A second chroma draw adds ≤ 0.001 dB,** so ×2 is the
  efficient setting.
- **Margins over supervised architectures grow with σ:** on BSD68, +0.07 / +0.10 / +0.14 dB over Restormer-gray. The
  same holds in colour ([`../diffusion_benchmark`](../diffusion_benchmark/README.md)).
- **Not yet run:** Urban100-gray, where Hi-IR / PTG-RM lead with 34.11–34.22 / 31.92–31.95 / 28.91. The GRL-B rerun
  is still running on a CPU kernel.

## Xformer on the same draws (the strongest released DFWB model at σ 25 / 50)

- **Run.** CPU kernel [`trnquanghuyn/gray-sota-rerun-xformer`](https://www.kaggle.com/code/trnquanghuyn/gray-sota-rerun-xformer)
  (1.6 h) of [`notebooks/gray_sota_rerun_xformer.ipynb`](../../notebooks/gray_sota_rerun_xformer.ipynb). Scores are in
  [`sota_reruns/`](sota_reruns/).
- **Setup.** Official σ-specific gray weights and test-script padding, seed 0.
- **Reproduction.** The rerun matches the paper to 0.005 dB in all six cells: BSD68 31.982 / 29.545 / 26.647 and Set12
  33.458 / 31.163 / 28.099.

Lift ×2 − Xformer on identical draws (dB; images better):

| | σ 15 | σ 25 | σ 50 |
|---|---:|---:|---:|
| BSD68 | +0.057 (68/68) | +0.078 (68/68) | +0.106 (68/68) |
| Set12 | +0.043 (11/12) | +0.049 (9/12) | +0.131 (11/12) |

## Released supervised gray models on the same draws

- **Run.** CPU kernel [`trnquanghuyn/gray-baselines-cpu`](https://www.kaggle.com/code/trnquanghuyn/gray-baselines-cpu)
  (166 min) of [`notebooks/gray_baselines_cpu.ipynb`](../../notebooks/gray_baselines_cpu.ipynb). Scores are in
  [`baselines_cpu/`](baselines_cpu/).
- **Models.** SCUNet-gray (KAIR v1.0, σ-specific, whole image, no x8) and DRUNet-gray (with a noise-level map).

Seed-0 rerun against published numbers (dB):

| | Set12 σ 15 / 25 / 50 | BSD68 σ 15 / 25 / 50 |
|---|---|---|
| SCUNet rerun | 33.426 / 31.089 / 28.039 | 31.891 / 29.386 / 26.325 |
| SCUNet published | 33.43 / 31.09 / 28.04 | 31.99 / 29.55 / 26.67 |
| DRUNet rerun | 33.245 / 30.936 / 27.896 | 31.826 / 29.335 / 26.306 |
| DRUNet published | 33.25 / 30.94 / 27.90 | 31.91 / 29.48 / 26.59 |

- **Set12 reproduces to 0.005 dB.**
- **BSD68 does not.** It is 0.08–0.35 dB low, more so at higher σ, although Restormer-gray reproduces on the same
  images to 0.005 dB. The likely cause is a different BSD68 copy in KAIR's evaluation [INFERENCE]. The same pattern
  appeared for DRUNet on CBSD68 / McMaster in [`../diffusion_benchmark`](../diffusion_benchmark/README.md).
- **Comparisons.** BSD68 comparisons with SCUNet / DRUNet therefore use their published numbers in the tables. The
  paired differences below are on identical draws.

Lift ×2 − baseline, paired over seeds 0–2 (dB; images better / pairs):

| | Set12 σ 15 | 25 | 50 | BSD68 σ 15 | 25 | 50 |
|---|---:|---:|---:|---:|---:|---:|
| − SCUNet-gray | +0.078 (36/36) | +0.124 (36/36) | +0.190 (36/36) | +0.149 (204/204) | +0.235 (204/204) | +0.431 (204/204) |
| − DRUNet-gray | +0.254 (36/36) | +0.280 (36/36) | +0.342 (36/36) | +0.214 (204/204) | +0.289 (204/204) | +0.451 (204/204) |

## Per-image R2R adaptation on top of the lift (σ 50): negative

- **Run.** Kaggle [`trnquanghuyn/gray-lift-r2r`](https://www.kaggle.com/code/trnquanghuyn/gray-lift-r2r) (166 min) of
  [`notebooks/gray_lift_r2r.ipynb`](../../notebooks/gray_lift_r2r.ipynb). Scores are in [`r2r_sigma50/`](r2r_sigma50/).
- **Settings.** All frozen from the Urban100 development images, where the same adaptation gained +0.127 dB at σ 50
  ([`../adaptation/README.md`](../adaptation/README.md)).

| | n | lift ×2, unadapted | adapted | gain | images worse by > 0.05 dB |
|---|---:|---:|---:|---:|---:|
| BSD68 | 68 | 26.754 | 26.535 | −0.219 | 93 % |
| Set12 | 12 | 28.231 | 27.044 | −1.187 | 92 % |

**Why it fails [INFERENCE].** For a fixed noisy image, the R2R target y − σz/α still contains that image's noise. A
gray BSD68 image has 154 k noisy values and a Set12 image 65–262 k, against 216 k adapted parameters, so the gains
can fit the noise. Urban100 colour images have about 2.4 M noisy values.

## GRL-B on the same draws (the strongest released gray model at σ 15)

Run: CPU kernel [`trnquanghuyn/gray-sota-rerun-grl`](https://www.kaggle.com/code/trnquanghuyn/gray-sota-rerun-grl)
(10.6 h, deadline-limited) of [`notebooks/gray_sota_rerun_grl.ipynb`](../../notebooks/gray_sota_rerun_grl.ipynb). Scores
are in [`sota_reruns/grl_scores.csv`](sota_reruns/grl_scores.csv).

- **Reproduction.** On identical seed-0 draws, GRL-B scores Set12 28.038 at σ 50 against its published 28.03, a
  +0.008 dB gap. The rerun is valid.
- **Paired result, Set12 σ 50.** The lift ×2 scores 28.230 against GRL-B's 28.038: **+0.192 dB**, with 11 of 12 images
  better (worst −0.047), sign-flip p = 9.3·10⁻⁴. That clears the pre-registered bar of +0.05 dB, p < 0.01, and ≥ 65 % of
  images better.
- **BSD68 is incomplete.** The deadline was reached after 4 images (26.651 against 26.60 published), so no BSD68 claim
  is made against GRL-B. The rerun is CPU-infeasible: at 50 min per image, 68 images need about 57 h.
- **The BSD68 copy is nevertheless validated, by two independent models.** On this repository's BSD68 (cszn/DnCNN
  `testsets/BSD68`), Restormer and Xformer reproduce their published numbers at every σ:

  | model | BSD68 σ 15 / 25 / 50 (rerun − published) | Set12 σ 15 / 25 / 50 |
  |---|---|---|
  | Restormer | +0.005 / +0.004 / −0.003 | −0.004 / −0.003 / 0.000 |
  | Xformer | +0.002 / −0.005 / −0.003 | −0.002 / +0.003 / −0.001 |
  | SCUNet | **−0.099 / −0.164 / −0.345** | −0.004 / −0.001 / −0.001 |
  | DRUNet | **−0.084 / −0.145 / −0.284** | −0.095 / −0.084 / −0.004 |

  - Two models reproducing to 0.005 dB rules out a copy or protocol error on our side.
  - SCUNet and DRUNet reproduce on Set12 and fail only on BSD68, so their published BSD68 numbers come from a
    different harness or crop, not from a different dataset copy.
  - **Consequence for the claims:** the paired margins over SCUNet / DRUNet on BSD68 (+0.43 / +0.45 dB) are measured
    against under-performing reruns and are **not claimed**. The Xformer comparison (68/68 images) rests on a validated
    rerun, and the comparison with GRL-B's published 26.60 is valid because the copy is validated.
- **Draw identity, seeds 1 and 2.** The seed-1 and seed-2 draws were re-derived single-threaded and match the lift job's
  Restormer scores on all 480 (image, σ, seed) pairs, with a largest difference of 0.0001 dB
  ([`draw_check/seeds12.csv`](draw_check/seeds12.csv)). The cross-job counts for those seeds therefore rest on
  identical draws.
