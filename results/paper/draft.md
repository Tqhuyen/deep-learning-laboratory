# Draft: a frozen diffusion prior for Gaussian denoising — what it buys, measured

Status: working draft. It was reframed after the critical-council review of 2026-11-06. The factual corrections of the
Phase-1 deconstruction (same day) are applied; mechanism claims marked "not run" or "not separated" are not established. Every number
is taken from a CSV in this repository. The colour study is in
[`../diffusion_benchmark/README.md`](../diffusion_benchmark/README.md) and the gray study in
[`../gray_lift/README.md`](../gray_lift/README.md).

## Abstract

A pretrained pixel-space diffusion model is a strong denoiser: one evaluation at the step matched to the noise level
gives its posterior mean. We measure how far that estimate goes on the standard Gaussian benchmarks, what it costs
against the published state of the art, and where it fails. Every claim is stated in the form that survives a paired,
same-image test, because the scoring protocol turns out to move a fixed denoiser by more than the margins being
ranked.

- **Colour, standard sizes.** On CBSD68, Kodak24 and McMaster, one step plus its horizontal flip is within 0.02 dB of
  DMID-d's 1500-evaluation ensemble. At σ = 50 it stays below DMID-d on every set. DMID's own ablation already reports
  that one step is best without repetitions, so this is a re-measurement. Calibration, ensembles of five networks,
  class mixtures and fine-tuning on DIV2K each add at most 0.02 dB.
- **Gray.** A lift with isotropic chroma noise (DDRM's spectral construction for a channel-mean operator, applied in
  one step) makes an RGB prior usable on gray images. It gives the first diffusion-prior results on BSD68 and Set12:
  32.04 / 29.62 / 26.75 dB on BSD68 at σ = 15 / 25 / 50. On identical noise it beats Xformer on all 68 BSD68 images
  at every σ. Xformer is the strongest released DFWB network at σ = 25 / 50; GRL-B, ahead at σ = 15, is being rerun.
- **Large images: a measured win.** Urban100's full 100 images, with settings frozen beforehand:
  - 384-px tiles plus 90 steps of per-image adaptation give 33.402 / 30.671 dB at σ = 25 / 50.
  - Paired against GRL-B's *released outputs* on the same 100 images and the same clean copies, that is
    **+0.065 dB** (95 % CI +0.034 to +0.096, 73/100 images) and **+0.220 dB** (CI +0.183 to +0.257, 92/100).
  - The σ = 50 margin is replicated on two further noise draws: the gain is **+0.1517 ± 0.0027 dB** across three
    independent draws (263 of 264 image–draw pairs better).
  - GRL-B is the strongest supervised denoiser without an external prior that releases weights.
  - At σ = 15 there is no gain (−0.031 dB, CI covers zero), although the adaptation itself still gains +0.134 dB on a
    cleanly held-out set.
- **Why earlier "wins" were not real.** A measurement audit shows that scoring a *fixed* denoiser on cropped versus
  full images, or on the Y channel versus RGB, moves its PSNR by 0.05–0.17 dB and by 1.5–2.6 dB. On Kodak24 the
  published ordering of two methods reverses when both are scored on the same pixels. SOTA margins of 0.05 dB are
  therefore only meaningful in a paired, same-image form, which is what the Urban100 result above uses.
- **Per-image adaptation (test-time training), and why it works where it does.** Recorrupted-to-Recorrupted tuning of
  0.04 % of the network on the noisy image alone adds +0.15 dB on Urban100, improving all 88 held-out images, and loses
  on Kodak24, McMaster and gray images. Whether it helps is a reproducible property of the *image* (per-image gain
  ranks identically across noise draws, ρ = 0.98) and is predicted before adapting by one measured quantity, the prior's
  context deficit (ρ = −0.44 to −0.56 within each σ). Deficit and gain rank identically across σ (from −0.082/+0.134 at
  σ = 15 to −0.187/+0.148 at σ = 50) and across datasets, which also explains the one cell without a win.
- **Limit.** A supervised network refined with CLIP features (PTG-RM) remains 0.1–0.3 dB ahead in most cells.

## 1. Introduction

A denoiser is the simplest test a generative prior can face, and the most brutal. The task has one number to report,
state-of-the-art margins on the standard benchmarks have shrunk to hundredths of a decibel, and the published
leaderboard is assembled from papers that each score their own copy of the data.

This paper takes one frozen, publicly available pixel-space diffusion model — the 256×256 unconditional ADM trained on
ImageNet — and asks a narrow question: **what can it actually do on Gaussian denoising, measured so that the answer
survives a skeptical reader?**

Three findings turned out to matter, and one of them is uncomfortable for the framing we started with.

**The prior is competitive but not magic.** One network evaluation, placed at the diffusion step whose noise level
matches σ, is within 0.02 dB of DMID's ensemble of 500 samples of a three-evaluation chain, i.e. 1500 evaluations
(Section 6.3). That is a factor of 750 in compute for 0.03 dB. The same evaluation is also within ±0.05 dB of a
supervised SwinIR trained on 100 k ImageNet images (Klug & Heckel, ICLR 2023), which tells us where the advantage over
ordinary supervised denoisers comes from: data scale, not diffusion. And it is 0.1–0.3 dB below a 2024 method that feeds
CLIP features into a supervised backbone. The frozen prior is a strong denoiser, not a new kind of denoiser.

**Two structural problems have to be fixed before it is usable.** Images far larger than the training resolution give
the model about twelve times the attention token count it was trained on, and grayscale images give it a channel
structure it never saw. For the first, overlapping tiles at 1.5–2× the training size recover part of the loss, and this
is not an artifact of averaging — 84–88 % of the gain survives on pixels covered by a single tile. For the second, an
exact lift with isotropic chroma noise makes the RGB prior's likelihood exactly isotropic, which turns a ten-year-old
impossibility into a two-evaluation operation and yields, to our knowledge, the first diffusion-prior numbers on Set12
and BSD68.

**Per-image test-time adaptation, and a measured limit.** Tuning 0.04 % of the network on the noisy image alone, with a
Recorrupted-to-Recorrupted loss and no clean data, adds about 0.15 dB on large, self-similar images and *subtracts*
0.02–1.2 dB on smaller ones. On the full 100 Urban100 images, with settings frozen beforehand, the pipeline reaches
33.402 / 30.671 dB at σ = 25 / 50. Paired against GRL-B's released outputs on the same 100 images and the same clean
copies — the strongest supervised denoiser without an external prior that publishes weights — that is **+0.065 dB** and
**+0.220 dB**, with 73 and 92 of 100 images better (Section 6.2); the σ = 50 margin reproduces on two further noise
draws (+0.1517 ± 0.0027 dB on 88 held-out images). At σ = 15 there is no gain, and we say so.

**Both fixes are the same fix.** Whether adaptation helps turns out to be a reproducible property of the *image* — the
per-image gain ranks identically across noise draws (ρ = 0.98) — and it is predicted *before* adapting by how much the
frozen prior's context mismatch costs that image (ρ = −0.62 on 88 images; −0.54 after controlling for baseline
quality). Tiling and adaptation substitute for each other. Urban100's context deficit is 0.19 dB, Kodak24's and
McMaster's 0.02–0.03 dB, and those are exactly the two sets where adaptation loses. Two forward passes therefore decide
whether to spend 4.3 GPU-minutes.

**Why the comparison had to be paired.** Assembling these numbers required an audit that became a contribution in its
own right. Holding a single denoiser fixed and changing only the scoring rule moves its PSNR by 0.05–0.17 dB through
cropping and border handling, and by 1.5–2.6 dB through the choice of colour channel. Two published methods' ordering on
Kodak24 reverses when both are scored on the same pixels. Since recent SOTA margins are 0.00–0.11 dB, a cross-paper
comparison at that scale is not measuring the method. This is also why the paper makes no claim of the form "we beat the
published number X", and why the one place we do claim a win is scored by one code base on one set of images.

**Contributions.**

1. A measured, paired statement of what a frozen diffusion prior achieves on Gaussian denoising, with every claim in
   [`claims.md`](claims.md) carrying its evidence and its falsification condition.
2. Evidence that the prior's edge over supervised denoisers is data scale: a supervised SwinIR at ImageNet scale matches
   it within 0.05 dB.
3. The first diffusion-prior results on grayscale Set12 / BSD68, and a paired win over every released supervised gray
   denoiser on identical draws.
4. A per-image test-time adaptation that gains +0.065 / +0.220 dB on Urban100 at σ = 25 / 50 against GRL-B's released
   outputs, replicated on three noise draws, plus a two-pass predictor of when it will help — which explains, rather
   than merely reports, its failure on small images.
5. A measurement of how much the scoring protocol costs, which bounds what any leaderboard claim below ~0.15 dB can
   mean.

**What we do not claim.** State of the art on CBSD68 / Kodak24 / McMaster, or on Urban100 at σ = 15; anything against
PTG-RM, which releases no weights; "training-free", since the pipeline performs test-time training; or real-noise
state of the art. The full list is in [`claims.md`](claims.md).

### 1.1 Naming and scope

- **"Training-free" is not used.** ADM's ε objective is denoising score matching on ImageNet-1k, so the model is a
  supervised Gaussian denoiser trained on about 150× more images than the DFWB baselines. And the per-image adaptation
  of Section 4.3 is test-time training, by name and by cost.
- **The class this paper works in** is therefore: a generic pretrained generative prior, no task-specific denoising
  training, plus per-image test-time adaptation.
- **The class its comparisons live in** is: released weights only. PTG-RM (CVPR 2024) holds the best published numbers
  in most cells but releases no denoising weights, so it cannot be rerun and cannot be entered into a paired test. Its
  numbers appear in this paper once, as a reference column.
- **Everything else is measured here**, on this repository's copies, with the noise draws stored and re-derivable.

## 2. Related work

**Supervised Gaussian denoisers.**
- DnCNN and FFDNet; DRUNet (DPIR).
- Transformers: SwinIR, Restormer, GRL, Xformer, HAT and Hi-IR.
- State-space models: MambaIR, MambaIRv2 and MaIR.
- PTG-RM (CVPR 2024) refines a supervised backbone with CLIP features. It holds the best published numbers in nearly
  every cell but releases no denoising weights.

Best published colour PSNR at σ = 15 / 25 / 50, verified against the papers' tables:

| set | supervised, no external prior | supervised + CLIP (PTG-RM) | training-free |
|---|---|---|---|
| CBSD68 | 34.48 MambaIR / 31.86 MaIR / 28.66 MaIR | 34.76 / 32.07 / 28.94 | 34.45 / 31.86 / 28.72 (DMID-d) |
| Kodak24 | 35.53 / 33.09 / 30.04 (MaIR) | 35.72 / 33.27 / 30.21 | 35.51 / 33.12 / 30.14 (DMID-d) |
| McMaster | 35.73 (GRL-B, MambaIRv2) / 33.46 (GRL-B) / 30.42 (Hi-IR) | 35.96 / 33.75 / 30.63 | 35.72 / 33.49 / 30.50 (DMID-d) |
| Urban100 | 35.54 / 33.35 (GRL-B) / 30.60 (IPT-V2; Hi-IR 30.59) | 35.70 / 33.57 / 30.64 | 35.26 / 33.11 / 30.28 (DMID, one step) |

- **Self-ensemble.** MaIR+ adds 0.03–0.04 dB, e.g. Kodak24 35.56 / 33.13.
- **LCDD (CVPRW 2025)** reports 34.48 / 31.90 / 28.77 on CBSD68 for a one-step ADM schedule. It scores four 256×256
  crops per image, so its numbers are not comparable with whole-image results. Its λ = 0.2 / 0.6 rows are implausible:
  about 4 dB above both of their endpoints.
- **Gray:** see [`../gray_lift/README.md`](../gray_lift/README.md).
  - Without an external prior: BSD68 32.00 GRL-B / 29.55 Xformer / 26.67 SCUNet; Set12 33.49 HAT / 31.19 Hi-IR /
    28.15 Hi-IR.
  - PTG-RM: 32.29 / 29.81 / 26.91 and 33.74 / 31.30 / 28.37.
  - No diffusion-prior method has reported gray results. The training-free classics do: BSD68 31.07 / 28.57 / 25.62
    (BM3D) and 31.37 / 28.83 / 25.87 (WNNM), as reported in the DnCNN paper.

**Diffusion models for denoising.**

* **DMID** (TPAMI 2024) embeds the noisy image at the matched step of a frozen ADM. DMID-d then averages 500
  samples of a three-evaluation chain (1500 evaluations per image); on Urban100 it uses one step. DMID-p samples for
  perception.
* **LCDD** (CVPRW 2025) linearly combines a one-step and a multi-step estimate to trade distortion for perception.
* **Earlier diffusion restoration methods** sample a full chain for general inverse problems and are weaker in PSNR on
  denoising: DDRM, DDNM, DPS, DiffPIR and GDP.

**Prior art for each component.**

* **The gray lift.** It is DDRM's (Kawar et al. 2022) spectral noise top-up for colourisation (H = channel mean,
  $s = 1/\sqrt3$) at the matched level.
* **One-step posterior mean.** Kawar et al. (2021) use the score network at the matched step as an MMSE denoiser.
* **Attention rescaling for unseen resolutions.** Our κ is Jin et al. (NeurIPS 2023), Eq. 7.
* **A 256-px prior on larger images.** Cheng & Tan (NeurIPS 2024) use ADM-256 as a "local" prior on larger images and
  find it better than ADM-512.
* **Overlapping patch inference for diffusion restoration.** WeatherDiffusion, StableSR, and patch-based diffusion
  priors (Hu et al. 2024).
* **Per-image gain adaptation.** GainTuning (Mohan et al., NeurIPS 2021) adapts per-channel gains on one test image.
  It gains most on Urban100's repetitive images.
* **Single-image test-time adaptation of denoisers.** LAN (CVPR 2024), TTAD (ICCV 2025), SCD and DDIP.

**Test-time and calibration methods.** These choose a denoiser's hyper-parameters without ground truth: Monte-Carlo
SURE (Ramani et al. 2008), and SURE for diffusion MRI reconstruction (SMRD, 2023). In our measurements SURE-weighted
aggregation of supervised denoisers adds ≤ 0.06 dB (CPU check on CBSD68 crops); it is not pursued.

**How this literature is evaluated, and what has been audited.**

- **Leaderboards rank on single draws.** The NTIRE σ 50 reports rank teams on a hidden 200-image set with one noise
  draw and no confidence intervals. The 2026 report's top four sit within 0.03 dB and are still ranked.
  [[NTIRE 2025]](https://arxiv.org/abs/2504.12276) [[NTIRE 2026]](https://arxiv.org/abs/2606.16031)
- **Variability is rarely reported.** A survey of 221 MICCAI-2023 segmentation papers found more than half report no
  variability, with a median confidence interval three times the gap between the top two methods
  (Christodoulou et al., MICCAI 2024). No such survey exists for low-level vision.
- **Training seeds have been varied, but not protocols.** A low-dose CT benchmark varied training seeds across eight
  methods with Mann–Whitney tests and found no method consistently ahead (Eulig et al., Medical Physics 2024); it
  explicitly treated natural-image denoising as having a consensus benchmark.
- **Contamination of large pretraining corpora is measured elsewhere.** Ramos et al. (ICCVW Findings 2025) find 4.64 %
  of the ImageNet test set in LAION, but do not examine BSD68, Kodak24, McMaster or Urban100.
- **Benchmarks of diffusion plug-and-play solvers** exist for scientific inverse problems (InverseBench, ICLR 2025) and
  report mean ± SD over test cases, but deliberately exclude natural-image denoising.
- **What is missing** is the measurement this paper makes: holding one denoiser fixed and changing only the scoring
  rule, against the margins being ranked. That is 6.5.

## 3. Background: the one-step posterior mean

Let $y = x + \sigma n$, $n \sim \mathcal N(0, I)$, with images in $[0, 1]$. ADM works in $[-1, 1]$ with
$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$.

* **Embedding.** Set $x_0 = 2x - 1$ and $\bar\alpha = 1/(1 + 4\sigma^2)$. Then
  $\sqrt{\bar\alpha}\,(2y - 1) = \sqrt{\bar\alpha}\,x_0 + \sqrt{1-\bar\alpha}\,n$. The scaled noisy image *is* a sample
  of the forward process at the step $t^*$ with $\bar\alpha_{t^*} = \bar\alpha$, with $\varepsilon = n$.
* **One step.** Tweedie's formula gives the posterior mean in one network evaluation:
  $\hat x_0 = (x_{t^*} - \sqrt{1-\bar\alpha}\,\hat\varepsilon(x_{t^*}, t^*))/\sqrt{\bar\alpha}$.
* **Sampling and PSNR.** An exact posterior sample has twice the MSE of the posterior mean, and averaging samples
  approaches the mean. DMID-d's 500-sample average is still 0.014 / 0.031 dB above one step at σ = 25 / 50. The
  horizontal-flip average (two evaluations) closes that gap. In our earlier measurement, averaging K reverse-chain
  samples converges as a + b/K, with the limit a within 3 % of the one-step MSE
  ([`../jepa_guided_diffusion/README.md`](../jepa_guided_diffusion/README.md), Section 9).

## 4. Method

Three changes to the one-step posterior mean of Section 3. Each was checked by a pre-registered test before it was
adopted (Appendix: decision log).

**4.1 Gray images: the isotropic chroma-noise lift** ([`../gray_lift/README.md`](../gray_lift/README.md)).
- **Lift.** For a gray observation $y = x + \sigma n$, form $Y = y\mathbf 1 + w$ with
  $w \sim \mathcal N(0, 3\sigma^2(I - uu^\top))$ and $u = \mathbf 1/\sqrt3$. The noise in $Y$ is then exactly
  $\mathcal N(0, 3\sigma^2 I)$, the forward process of the RGB prior at $\sqrt3\sigma$, and the luminance is unchanged.
- **Estimate.** The gray estimate is the channel mean of the one-step posterior mean at $\sqrt3\sigma$. Averaging the
  chroma draw (K = 2) adds ≤ 0.001 dB; the horizontal flip adds 0.011–0.021 dB.
- **Replication baseline.** Replicating $y$ into three channels gives noise that is perfectly correlated across
  channels; stepped at σ it is 5–9 dB worse. That baseline is conservative in the lift's favour: the luminance noise of
  a replicated image is √3σ, so the fair version steps at √3σ. **Not run.** The measured margin over the replicate
  baseline should therefore be read as an upper bound on the value of the lift.

**4.2 Large images: tiles near the training resolution** ([`../urban100/README.md`](../urban100/README.md)).
- **Problem.** ADM-256 has attention at 1/8 resolution. On a 1024-px image that layer sees about 12× its training
  token count.
- **Fix.** Overlapping tiles blended with linear ramps.
  - 512-px tiles beat the whole image on the 88 held-out Urban100 images by +0.05 / +0.09 / +0.15 dB.
  - 384-px tiles are better still on the development images.
  - Rescaling the attention logits by $\sqrt{\log N/\log n_{\rm train}}$ (Jin et al. 2023) recovers about half of
    that gain.
  - Overlap averaging is not the cause. On pixels that get a single tile estimate per pass, the gain is still 84–88 %
    of the all-pixel gain (88/88 images at σ = 25 / 50). **Not separated:** attention token count versus local
    GroupNorm statistics versus the convolutional receptive field. The attention-scaling arm recovers about half, which
    bounds the attention share from below.
  - On Kodak24, 256-px tiles lose, so the optimum lies above the training size.

**4.3 Per-image self-supervised adaptation** ([`../adaptation/README.md`](../adaptation/README.md)).
- **Parameters.** One gain per output channel of every 2-D convolution plus every GroupNorm affine: 0.2 M parameters,
  0.04 % of ADM. The attention block's 1-D qkv / projection convolutions are not adapted.
- **Loss.** Recorrupted-to-Recorrupted on the test image alone: input $y + \alpha\sigma z$ at the step matched to
  $\sigma\sqrt{1+\alpha^2}$, target $y - \sigma z/\alpha$, with $\alpha = 0.5$.
  - In expectation this is the clean MSE at the slightly higher level.
  - The time-conditioned network transfers the adaptation to the step matched to $\sigma$.
- **Training.** 90 steps of four 256² crops, Muon (AdamW branch for these 1-D parameters), lr $10^{-3}$. This costs
  about 4.3 GPU-minutes per image on a T4, including scoring.
- **Development results.** On the 12 Urban100 development images it gains +0.110 / +0.127 dB at σ = 25 / 50 after 90
  steps. A clean-target oracle, run for 60 steps at σ = 25 only, gains +0.172.
  - At matched 60 steps R2R reaches 55 % of the oracle, below the pre-registered 60 % gate. The gate was passed by
    comparing 90 R2R steps with 60 oracle steps.
  - The adaptation is image-specific: parameters transferred to another image lose 0.10 dB.
- **Where it fails, and why.** On gray BSD68 / Set12 at σ = 50, the same loss in the gray domain, with a lifted input,
  *loses* 0.22 / 1.19 dB. It also loses on Kodak24 (−0.02) and McMaster (−0.09) at σ = 50.
  - **Not size.** Inside Urban100's 88 held-out images the gain does not grow with image size (Spearman −0.10 / −0.12),
    so the "noisy values per parameter" rule of an earlier draft is withdrawn.
  - **Not baseline quality.** Controlling for the context deficit below, baseline PSNR has no independent effect
    (partial ρ = +0.04, p = 0.74).
  - **The context deficit.** Tiling and adaptation both act on the prior's mismatch with the image, and they substitute
    for each other. Measured before any adaptation, the context deficit `PSNR(whole ×1) − PSNR(512-tiles ×2)` predicts
    the adaptation gain with **ρ = −0.62** over 88 images (partial ρ = −0.54 controlling for baseline PSNR, p = 6·10⁻⁸),
    and reproduces at σ = 25 (ρ = −0.54).
  - **Across noise levels.** The deficit grows with σ (−0.082 / −0.119 / −0.187 dB at σ = 15 / 25 / 50) and the
    adaptation gain grows with it (+0.134 / +0.142 / +0.148 dB); per-image ρ is −0.44 / −0.52 / −0.56 on 100 images
    each, and the two rank identically across σ. **This is also the explanation of the σ = 15 result:** the prior's
    deficit is smallest there, so adaptation has least to recover, and that is why the pipeline does not overtake GRL-B
    at σ = 15 while it does at the two higher levels. No separate explanation is required.
  - **Across sets.** Urban100's deficit is −0.19 dB at σ = 50, against −0.03 (Kodak24) and −0.02 (McMaster); the rank
    correlation between deficit and adaptation gain over the four measured cells is −1.0. The two sets with a deficit
    below 0.03 dB are exactly the two where adaptation loses.
  - **A cheap pre-check.** The deficit costs two forward passes, so whether to adapt can be decided before paying its
    4.3 GPU-minutes.

**Tested and rejected: fine-tuning on the denoising steps** ([`../adm_finetune/README.md`](../adm_finetune/README.md)).
ADM was fine-tuned with its own ε loss on embedded noisy DIV2K crops (σ ~ U[5, 60]/255). By Section 3 this loss is the
one-step estimate's MSE up to a 4σ² factor.

* **Result:** after 1,678 steps (AdamW 2·10⁻⁵, effective batch 8), the test sets moved by −0.001 / +0.001 /
  +0.001 dB pooled over σ = 15 / 25 / 50.
* **Direction of the shift:** slightly up on Kodak24 and slightly down on CBSD68. On Urban100 it was up by
  0.02–0.03 dB, but only on the 4–7 smallest images: the rest ran out of memory, so that number is size-biased.
* **Validation:** DIV2K validation flattened at +0.01 dB within 400 steps.

**Tested and rejected: class-mixture posterior means and ensembles of different pretrained networks**
([`../prior_ensembles/README.md`](../prior_ensembles/README.md)).

* **Mixtures:** $E[x \mid y] = \sum_c p(c \mid y)\,E[x \mid y, c]$, with OpenAI's noise-aware classifiers and
  class-conditional ADMs at 256 and 512. The mixture is 0.012–0.023 dB below the unconditional flip average; the 512
  model is 0.1 dB below it.
* **Ensembles:** an equal-weight ensemble of five networks gains ≤ 0.004 dB.

## 5. Experimental protocol

* **Test sets:** CBSD68, Kodak24, McMaster and Urban100 from DMID's release.
* **Noise:** σ ∈ {15, 25, 50}/255. Noise is normalised to a sample standard deviation of exactly σ/255 (DMID's
  protocol) and never clipped, with fixed seeds, so every method we run sees identical inputs. DMID-d's released
  outputs use their own noise draws. Per-image paired tests against them are unbiased, and with noise normalised to
  exactly σ the extra draw variance is small. One step's deficit at σ = 25 / 50 is significant (Holm p < 0.001).
* **Gray:** Restormer's protocol (per-image `np.random.seed`, unnormalised); seed 0 is the published one.
* **Inference:** the whole image, reflect-padded to a multiple of 64; for large images, overlapping tiles (4.2).
* **Metrics:** PSNR on uint8 for every result. SSIM and LPIPS (AlexNet) are reported for the colour benchmark and real
  noise only; the Urban100 adaptation and the gray reruns recorded PSNR only.
* **Statistics:** paired bootstrap intervals and sign-flip tests per image, with Holm correction.
* **Reproduction checks:** DMID's released outputs, rescored with our code, reproduce its Table 1 to 0.005 dB.

## 6. Results

### 6.0 Summary table

Everything the paper claims, next to the published bars it is measured against. "Paired" means both methods were scored
by this repository's code on the same images; "reference" means the number is a published value on someone else's copy
and is shown only for orientation.

**Colour, PSNR dB (σ = 15 / 25 / 50).**

| set | ours | best published, no external prior | PTG-RM (CLIP, no weights) | our margin (paired) |
|---|---|---|---|---|
| CBSD68 | 34.462 / 31.858 / 28.704 | 34.50 / 31.88 / 28.69 (MaIR+) | 34.76 / 32.07 / 28.94 | — |
| Kodak24 | 35.522 / 33.117 / 30.138 | 35.56 / 33.13 / 30.08 (MaIR+) | 35.72 / 33.27 / 30.21 | — |
| McMaster | 35.734 / 33.493 / 30.487 | 35.74 / 33.48 / 30.42 (MaIR+/Hi-IR) | 35.96 / 33.75 / 30.63 | — |
| **Urban100** | **35.501 / 33.402 / 30.671** | 35.54 / 33.35 / 30.60 (GRL-B / IPT-V2) | 35.70 / 33.57 / 30.64 | **−0.031 / +0.065 / +0.220** |

**Gray, PSNR dB (σ = 15 / 25 / 50), identical draws.**

| set | ours (lift ×2) | best released, rerun here | PTG-RM (published) | paired result |
|---|---|---|---|---|
| BSD68 | **32.039 / 29.623 / 26.753** | Xformer 31.982 / 29.545 / 26.647 | 32.29 / 29.81 / 26.91 | Xformer: 68/68 images every σ |
| Set12 | **33.501 / 31.211 / 28.230** | Xformer 33.458 / 31.163 / 28.099; GRL-B 28.038 at σ 50 | 33.74 / 31.30 / 28.37 | Xformer 107/108 pairs; GRL-B σ 50 +0.192 |

**Where the method does not win.**

| cell | outcome |
|---|---|
| CBSD68 / Kodak24 / McMaster, σ 50 | one step ×2 is 0.01–0.03 dB *below* DMID-d, and 0.07–0.26 dB below PTG-RM |
| Urban100 σ 15 | −0.031 dB against GRL-B, confidence interval covering zero |
| Real noise | parity with DMID-d only; below Cheng & Tan and P2N |
| Per-image adaptation on small and gray images | −0.02 to −1.9 dB |


**6.1 Gray (Set12, BSD68), Restormer's protocol, seed 0** ([`../gray_lift/README.md`](../gray_lift/README.md)).

| method | training data | Set12 σ 15 / 25 / 50 | BSD68 σ 15 / 25 / 50 |
|---|---|---|---|
| SCUNet (published) | DFWB | 33.43 / 31.09 / 28.04 | 31.99 / 29.55 / 26.67 |
| Xformer (published) | DFWB | 33.46 / 31.16 / 28.10 | 31.98 / 29.55 / 26.65 |
| GRL-B (published) | DFWB | 33.47 / 31.12 / 28.03 | 32.00 / 29.54 / 26.60 |
| Hi-IR / HAT (published) | DFWB | 33.48 (Hi-IR) / 31.19 / 28.15 | — |
| PTG-RM, GRL-B + CLIP (published) | DFWB + CLIP | 33.74 / 31.30 / 28.37 | 32.29 / 29.81 / 26.91 |
| **ADM + lift ×2 (ours)** | ImageNet-1k (ADM's ε objective) | **33.501 / 31.211 / 28.230** | **32.039 / 29.623 / 26.753** |

- **Training data.** DFWB = DIV2K + Flickr2K + BSD400 + WED. Supervised rows use their published numbers; Hi-IR
  reports no BSD68 result.
- **Identical draws.**
  - Our reruns of Restormer-gray and Xformer reproduce their papers to 0.005 dB.
  - Seeds 0–2: the lift beats Restormer-gray on 204/204 BSD68 pairs and 35/36 Set12 pairs, by +0.08 to +0.23 dB. It
    beats SCUNet-gray on every pair.
  - Seed 0: it beats Xformer, the strongest released DFWB network at σ = 25 / 50, on 68/68 BSD68 images at every σ
    (+0.06 / +0.08 / +0.11 dB) and on 9–11 of 12 Set12 images.

**6.2 Colour Urban100** ([`../urban100/README.md`](../urban100/README.md),
[`../adaptation/README.md`](../adaptation/README.md), [`../urban100/full100`](../urban100/full100/README.md)).
- **Reproduction.** The whole-image one step reproduces DMID's released Urban100 outputs (35.263 / 33.108 / 30.281).
- **Tiles.** 512-px tiles ×2 give 35.345 / 33.227 / 30.468.
  - On the held-out 88 images, with identical draws and one evaluation per tile, they beat the whole image by +0.044 /
    +0.074 / +0.127 dB (84 / 88 / 88 images better); with the flip on both sides, by +0.05 / +0.09 / +0.15 dB.
  - Overlap averaging is not the cause: on pixels that get a single tile estimate per pass, 84–88 % of the gain
    remains.
- **Full 100 images, measured.** 384-px tiles ×2 plus 90 R2R steps, settings frozen on dev-12 beforehand:
  33.402 / 30.671 dB at σ = 25 / 50, and 35.501 dB at σ = 15.
  - The σ = 15 run was a genuine held-out test: nothing was tuned at that level. Working on confirm-88 it gains
    +0.138 dB, 88/88 images, p < 10⁻⁵.
- **Paired against GRL-B's released outputs.** Same 100 images, same clean copies, rescored by our code; GRL-B's
  released outputs reproduce its published numbers (35.532 / 33.337 / 30.450 against 35.54 / 33.35 / 30.46).

  | σ | ours | GRL-B (released) | difference | 95 % CI | images better | p |
  |---|---:|---:|---:|---|---:|---:|
  | 15 | 35.501 | 35.532 | −0.031 | [−0.072, +0.010] | 52/100 | 0.15 |
  | 25 | 33.402 | 33.337 | **+0.065** | [+0.034, +0.096] | 73/100 | 1·10⁻⁴ |
  | 50 | 30.671 | 30.450 | **+0.220** | [+0.183, +0.257] | 92/100 | < 10⁻⁵ |

  Both methods are scored by the same code on the same pixels, so the protocol effects of 6.5 cancel. What remains is
  the method and the noise draw; the paired interval absorbs the draw.
- **Replication at σ = 50.** Two fresh draws give +0.1545 / +0.1491 / +0.1515 dB on confirm-88 (draws 0 / 1 / 2), mean
  **+0.1517 dB** with a cross-draw SD of **0.0027 dB**, and 263 of 264 image–draw pairs better
  ([`../urban100/seed_replication`](../urban100/seed_replication/README.md)). σ = 25 was not replicated.
- **By-product: the draw effect is 0.002 dB, not 0.02.** Across the three draws the *unadapted* 88-image set mean moves
  by a standard deviation of 0.0017 dB. Comparisons across draws are therefore close to exact at the set level, and the
  0.014 / 0.031 dB deficit of one step against DMID-d at σ = 25 / 50 is real.
- **What it does not establish.** At σ = 15 there is no gain. PTG-RM has no released weights, so it cannot enter this
  comparison; its published numbers are 0.17 dB above ours at σ = 25 and 0.03 dB below at σ = 50 on its own protocol.
  The pipeline does test-time training, so the class is "no task-specific training plus test-time adaptation".

**6.3 CBSD68, Kodak24, McMaster.**
- **Near-parity.** One step ×2 is within 0.02 dB of DMID-d, and below it at σ = 50 (Section 7).
- **Tiles on the small sets** ([`../diffusion_benchmark/tile_rule`](../diffusion_benchmark/tile_rule/README.md)), rule R
  applied unchanged: Kodak24 +0.017 / +0.019 / +0.017, McMaster +0.010 / +0.011 / +0.002, CBSD68 −0.005 (σ 25) /
  −0.019 (σ 50). Tiling is a large-image tool only.
- **No adaptation gain.** Per-image R2R does not help on these smaller images: Kodak24 −0.02, McMaster −0.09 dB at
  σ = 50. Gray images lose 0.2–1.2 dB. The cause is not identified (4.3).
- **DMID's own multi-evaluation chain loses.** Its S_t = 2 chain at η = 0 (three evaluations per sample) scores
  −0.128 / −0.137 / −0.114 dB against one step at σ = 15 / 25 / 50, on 0 of 110 images at every σ. On matched Gaussian
  noise, one step is not merely sufficient; it is the best of the two.

**6.4 Real noise.** See 7b. These are DMID's own sampling ablations (its Tables 4–5) re-measured on DMID's transformed
images. Its chain with 16 flip-balanced samples ties DMID-d at 21–31× fewer evaluations. Cheng & Tan (NeurIPS 2024)
lead CC / PolyU and P2N (CVPR 2025) leads FMDD.

**6.5 The scoring protocol costs more than the margin**
([`../measurement_audit`](../measurement_audit/README.md)).

Holding one denoiser fixed (DMID-d's released full-image outputs) and changing only the scoring rule:

| what changes | effect on a fixed method's PSNR |
|---|---|
| crop to a multiple of 8 (GRL's protocol), Kodak24 | **−0.09 to −0.12 dB** |
| crop the last row and column (GRL's BSD68 / CBSD68) | +0.005 dB |
| shave 8 px off every border | −0.05 to +0.04 dB |
| **score the Y channel instead of RGB** | **+1.5 to +2.6 dB** |

Repeating the measurement with a second, architecturally different denoiser (Restormer, our own draws) shows the
effect belongs to the protocol, not to the model:

| axis | DMID-d | Restormer | largest disagreement |
|---|---|---|---|
| GRL crop, over 9 set–σ cells | −0.12 … +0.02 dB | −0.11 … +0.03 dB | **0.012 dB** |
| Y channel, over 9 set–σ cells | +1.57 … +2.79 dB | +1.58 … +2.81 dB | **0.024 dB** |
| **clipped, quantised input** | not measurable (its outputs are uint8) | −0.48 / −0.99 / −2.46 dB at σ = 15 / 25 / 50, up to −3.81 | new axis |

Two architectures agree on the first two axes to within 0.02 dB over nine cells, so the effect belongs to the protocol
rather than to the model.

The third row is new and is the largest axis of the three. Feeding the *same* method a clipped, quantised copy of the
same noisy image costs 0.48 dB on average at σ = 15, 0.99 dB at σ = 25 and **2.46 dB at σ = 50**, reaching **3.81 dB**
on McMaster. The mechanism is direct: clipping is a projection onto the valid range, so the fraction of pixels pinned to
the boundary grows with σ, and the more of the noise realisation is lost the more the reconstruction has to invent. A
3.8 dB term is 40× a recent SOTA margin, which means a clipped-protocol number and an unclipped one are not comparable,
and it also means that part of the NTIRE 2026 report's "+3.37 dB over public Restormer σ 50" is this convention rather
than a training effect.

And the ordering does not survive: on Kodak24 at σ = 15 and σ = 25, DMID-d is 0.08 and 0.10 dB *ahead* of GRL-B in the
published tables and 0.041 and 0.001 dB *behind* it when both are scored on the same pixels.

- **Consequence for this paper.** Every cross-paper margin below about 0.15 dB in this literature, including the ones
  this project previously quoted, is unresolvable unless both methods are rerun on the same images. The Urban100 result
  in 6.2 is reported in the only form that survives this: paired, same images, same code.
- **Consequence for the field.** Recent SOTA margins on these benchmarks are 0.00–0.11 dB. A leaderboard that mixes
  cropping rules, colour spaces and test-set copies cannot resolve them, and no top-venue audit of these protocols
  exists.

## 7. Analysis: what the remaining knobs add

From [`../diffusion_benchmark/README.md`](../diffusion_benchmark/README.md), paired over 110 images per σ:

| contrast (PSNR, dB) | σ 15 | σ 25 | σ 50 |
|---|---:|---:|---:|
| one step − DMID-d (1500 evaluations) | +0.001 | −0.014 | −0.031 |
| h-flip ×2 − one step | +0.011 | +0.013 | +0.016 |
| dihedral ×8 − ×4 | −0.000 | −0.002 | −0.007 |
| calibrated step − matched step | −0.000 | −0.000 | −0.000 |
| t → 0 → clean, two evaluations (η = 0) − one step | −0.023 | −0.014 | −0.006 |
| ×8 − Restormer | +0.079 | +0.101 | +0.134 |

The deterministic second step lowers LPIPS on every image (−0.008 / −0.010 / −0.013), the cheapest perception knob.

DMID's own S_t = 2 chain (three evaluations) at η = 0 *was* run on the small sets
([`../diffusion_benchmark/tile_rule`](../diffusion_benchmark/tile_rule/README.md)): it scores **−0.128 / −0.137 /
−0.114 dB** against one step at σ = 15 / 25 / 50, on 0 of 110 images at every σ. The chain that carries DMID-d on real
noise is therefore a loss on matched Gaussian noise, which sharpens Section 7's conclusion: on Gaussian noise, one step
is not a cheap approximation to DMID-d's estimate; it is the better estimate.

## 7b. Real noise

From [`../real_noise/README.md`](../real_noise/README.md): CC (15), PolyU (100) and FMDD (48). DMID's released outputs
rescore to its Table 3.

* **Pixel-shuffle decorrelation does not close the gap.**
  * Camera noise is still correlated at lag 2 (0.39–0.40).
  * PD4 plus the one-step posterior mean peaks at CC 36.29 / PolyU 37.88 / FMDD 30.86 dB.
  * DMID-d reaches 37.99 / 38.62 / 33.40 dB.
* **DMID-p is the one-step posterior mean of DMID's noise-transformed images.** Its per-image transformation does
  most of the work.
* **On the transformed images the chain, not the averaging, carries DMID-d.**
  * One sample of DMID's stochastic three-step chain is already 0.6–1.2 dB above one step on CC and FMDD.
  * 16 samples with half run on the flipped image (48 evaluations) tie DMID-d's 1,500/999 evaluations: paired
    +0.004 ± 0.008 (CC), −0.001 ± 0.002 (PolyU), +0.000 ± 0.004 dB (FMDD).
  * The sample average saturates (8 → 500 samples: ≤ 0.02 dB), so this stage cannot beat DMID-d. A gain in accuracy
    needs a better noise transformation.

## 8. Limitations

* **Not state of the art on every cell.** PTG-RM (supervised backbone + CLIP features, CVPR 2024) reports higher
  numbers in every colour cell and in gray Set12 / BSD68; it releases no denoising weights, so it cannot be rerun. Gray
  Urban100 σ = 50 is led by Hi-IR (28.91). A supervised SwinIR trained on 100 k ImageNet images (Klug & Heckel,
  ICLR 2023) matches ADM ×2 within ±0.05 dB at σ = 25 on all four colour sets, so ADM's edge over DFWB-trained networks
  is a training-data effect.
  The paired evidence that *is* claimed is limited to reruns of released models on identical data: GRL-B on Urban100
  (+0.065 / +0.220 dB at σ 25 / 50, Section 6.2), Xformer, Restormer, SCUNet and DRUNet on gray, and GRL-B on Set12
  σ 50. Unpaired margins over published numbers are not claimed anywhere in this paper.
* **Mechanisms are not established.**
  * The tile gain is not separated between attention token count and local normalisation statistics. Overlap
    averaging is ruled out as the main cause.
  * The adaptation's failure on small and gray images is not attributed.
  * The replication baseline for gray images uses a mismatched step.
* **Test-set overlap with ImageNet is unmeasured.** Urban100 was scraped from Flickr, and ADM was trained on ImageNet.
* **Research-only prior.** OpenAI's model card restricts the ADM checkpoints to research use ("not intended to be
  commercially deployed"); the MIT licence covers the code only. ADM was also trained on ImageNet. A deployable version
  would need a diffusion prior trained on licensed data.
* **Per-image adaptation is narrow.**
  * It is confirmed only on Urban100 at σ = 25 / 50.
  * It loses on Kodak24, McMaster and gray images.
  * It costs about 4 GPU-minutes per image on a T4.
* **Real noise is unsolved by this route.** Without DMID's per-image noise transformation, pixel-shuffle decorrelation
  stays 1.6 dB behind on CC. With it, the second stage is at parity with DMID-d but below Cheng & Tan (NeurIPS 2024)
  and P2N (CVPR 2025).
* **Protocol differences.**
  - Published supervised numbers use the KAIR protocol, where noise is not normalised to σ exactly. Our colour runs
    use DMID's protocol. DMID-d's released outputs use other draws of the same normalised noise.
  - Gray comparisons rerun Restormer, Xformer, SCUNet and DRUNet on identical draws. SCUNet / DRUNET do not reproduce
    their published BSD68 numbers on the DnCNN repository's BSD68, although Restormer and Xformer do. GRL-B was rerun on
    Set12 σ = 50 only (28.038 against 28.03 published); its BSD68 rerun was stopped by the CPU deadline after 4 images.
* **The claim is narrow, and this is deliberate.**
  * **Scope:** colour, Urban100, σ = 25 and 50. Not σ = 15 (−0.031 dB, CI covers zero). σ = 50 is replicated on three
    draws; σ = 25 is single-draw.
  * **Comparator:** GRL-B's released outputs, the strongest supervised model without an external prior that releases
    weights. PTG-RM has none.
  * **Form:** paired on the same 100 images with the same code. Not a comparison of published numbers.
  * **Class:** the pipeline performs test-time training, so it is not "training-free". The defensible class is
    "no task-specific training, plus per-image test-time adaptation".
  * **Protocol-free, not noise-free:** both methods use different draws; the paired interval absorbs that.
* **One denoiser supplies the protocol budget.** The 0.05–0.17 dB protocol shifts are measured on DMID-d alone. A
  second and third model would turn them into a budget rather than an illustration.
