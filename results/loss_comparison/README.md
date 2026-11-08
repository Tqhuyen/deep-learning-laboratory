# Does the loss matter? SURE against R2R at equal budget

The adaptation in [`../adaptation`](../adaptation/README.md) uses a Recorrupted-to-Recorrupted loss. Its closest prior
art, GainTuning (NeurIPS 2021), tunes the same kind of parameter — per-channel gains fitted on one test image — with a
**SURE** loss, and reports that it "does not require early stopping". We had never compared the two, and the comparison
on ADM needs GPU hours that were not available.

This study runs it on a denoiser that fits CPU.

- **Run.** CPU kernel `sure-vs-r2r-cpu` (2 h 16 min, stopped by its own safety gate on one hard image) of
  [`notebooks/sure_vs_r2r_cpu.ipynb`](../../notebooks/sure_vs_r2r_cpu.ipynb); `sure-vs-r2r-urban50` completes the
  Urban100 σ 50 cell.
- **Scores.** [`sure_vs_r2r.csv`](sure_vs_r2r.csv).

## Setup

- **Denoiser:** DnCNN-gray blind, KAIR release `dncnn_gray_blind.pth`. Its architecture was read off the checkpoint
  rather than assumed: **twenty 3×3 convolutions, every one with a bias, and no normalisation**.
- **Adapted parameters:** one gain per convolution output channel, 1,217 values, reset to 1 between cases. Identical in
  every arm.
- **Equal everything else:** 2 × 96² crops per step, 60 steps, Adam at lr 1e-3, same crop schedule, same noise draw.
- **Arms:**
  - `oracle` — MSE at the clean target, the headroom;
  - `r2r` — MSE(f(y + ασz), y − σz/α), α = 0.5;
  - `sure` — mean((f(y) − y)²) + 2σ²·div f, divergence by one Rademacher probe, relative ε = 1e-3.
- **Validity gate:** every case asserts the pretrained baseline exceeds 18 dB. The set means are 29.121 dB at σ 25 and
  26.069 at σ 50 on BSD68, against published 29.16 and 26.23, so the stand-in is at the published level.

## Result: R2R beats SURE in every cell

Gain over the unadapted baseline, dB:

| set, σ | n | oracle | **r2r** | **sure** | r2r − sure | r2r / oracle | r2r harmed | sure harmed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BSD68 σ 25 | 24 | +0.069 | **+0.036** | −0.100 | **+0.136** | 52 % | 0 % | 71 % |
| BSD68 σ 50 | 24 | +0.094 | **−0.070** | −0.135 | **+0.065** | −74 % | 46 % | 75 % |
| Urban100 σ 25 | 12 | +0.342 | **+0.262** | −0.055 | **+0.317** | 77 % | 0 % | 42 % |
| Urban100 σ 50 | 12 | +0.324 | **+0.183** | +0.003 | **+0.180** | 56 % | 0 % | 17 % |

72 cases in total. "Harmed" = worse than the unadapted baseline by more than 0.05 dB.

- **The pre-registered rule fires for R2R in all four cells:** it gains 0.065–0.317 dB more than SURE, far above the
  0.02 dB threshold, and it harms fewer images in every cell.
- **SURE is negative or zero everywhere.** With one probe per step the divergence term is a high-variance estimate of a
  quantity that grows as σ², and the adapted gains drift toward over-smoothing: 17–75 % of images end up worse than the
  frozen denoiser.
- **R2R's behaviour reproduces the ADM pattern on a different architecture.** It gains most on Urban100 (+0.26 / +0.18,
  56–77 % of the oracle) and loses on BSD68 at σ 50 (−0.070, 46 % of images harmed). That is the same set- and
  noise-level dependence reported in [`../adaptation`](../adaptation/README.md), where the explanation is the prior's
  context deficit: Urban100 has one, small high-noise images do not.

## What this does not settle, and it matters

**This is not GainTuning.** Their method uses their own SURE estimator (they average over noise realisations rather than
one projection per step), a learning-rate schedule dropped tenfold at step 20, 100 steps of 5,000 patches, and they
report that it almost never hurts. A single-probe Monte-Carlo SURE at 60 steps is a *weaker* instrument, and attributing
its failure to the SURE objective would be wrong.

What the experiment does establish is narrower and still useful:

1. **Our loss is not the weak link** among the choices we could implement at equal budget: on a different architecture,
   at a different scale, with a different parameter set size, R2R is the better of the two.
2. **The harm pattern transfers.** R2R loses on BSD68 at σ 50 (46 % of images harmed) exactly as it does on ADM and on
   the gray lift, which supports the claim in [`../adaptation`](../adaptation/README.md) that the failure is a property
   of the image and the noise level, not of the ADM architecture.
3. **Headroom exists.** The oracle reaches +0.07 to +0.29 dB with 1,217 gains, and R2R captures 52 % of it on BSD68
   σ 25 and 77 % on Urban100 σ 25 — but only 44 % on Urban100 σ 50 and none on BSD68 σ 50.

The experiment that would settle the GainTuning question — their estimator and schedule on ADM — still needs GPU hours
and is listed as not run in [`../paper/draft.md`](../paper/draft.md) §7c.

## Three failed attempts, and what caught them

Recorded because the first one is the kind of result that survives into a paper.

- **Run 1 (147 min) returned a clean null: every arm +0.000 dB on every set and σ.** The loader stripped only a
  `module.` prefix while the checkpoint uses `model.`, so the network stayed at random initialisation and no gain can
  rescue it. Caught by the baseline PSNR being 6.99 dB. A null result is exactly what a reader is least likely to
  question, and this one was an artefact.
- **Run 2** failed loudly on a shape mismatch, because the architecture had been guessed from the standard DnCNN sketch
  rather than read off the checkpoint.
- **Run 3** loaded the weights exactly but used the wrong input convention (0–255 instead of 0–1, and subtracting instead
  of adding the residual), scoring 19.65 dB. Caught by the gate.

The convention now in the code was established by measuring one image under all four candidate interpretations; the
correct one was the only one above the noise floor.
