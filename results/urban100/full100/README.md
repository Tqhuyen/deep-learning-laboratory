# Urban100, full 100 images: the frozen pipeline, measured

Run: Kaggle GPU kernel [`trnquanghuyn/urban100-full`](https://www.kaggle.com/code/trnquanghuyn/urban100-full)
(T4 ×2, 4 h 39 min) of [`notebooks/urban100_full.ipynb`](../../notebooks/urban100_full.ipynb). The settings were frozen
on dev-12 before this run.

Pipeline, unchanged from [`../adaptation`](../adaptation/README.md):

- **Inference:** 384-px tiles, overlap 96, horizontal-flip average (×2).
- **Adaptation:** per-channel conv gains + GroupNorm affine, Recorrupted-to-Recorrupted, α = 0.5, 90 steps,
  Muon (AdamW branch) at lr 1e-3.
- **Data:** Urban100 from DMID's release. Noise is DMID's protocol, normalised and unclipped.
- **Scores:** [`scores.csv`](scores.csv) per image, σ and arm; [`tests.csv`](tests.csv) the pre-registered tests.

## Results

| σ | unadapted | adapted | gain | images better | images worse by > 0.05 | n | role |
|---|---:|---:|---:|---:|---:|---:|---|
| 15 | 35.367 | **35.501** | +0.134 | 100/100 | 0 | 100 | **held-out test:** nothing was tuned at σ 15 |
| 25 | 33.260 | **33.402** | +0.142 | 100/100 | 0 | 88 + 12 | full 100 = confirm-88 plus this run's dev-12 |
| 50 | 30.522 | **30.671** | +0.148 | 100/100 | 0 | 88 + 12 | same |

- **σ 15, confirm-88 only** (the pre-registered test): 35.283 → 35.421, **+0.138 dB**, 88/88 images better, sign-flip
  p < 10⁻⁵. Success.
- **Full-100 paired gains:** σ 25 +0.1425 dB and σ 50 +0.1481 dB over 100 images, minimum per-image gain +0.009 /
  +0.007 dB, sign-flip p < 10⁻⁵.
- **dev-12 rows** (now that they exist) are descriptive only: they chose the settings.

## Paired against GRL-B's released outputs, same 100 images

GRL-B released its Urban100 denoised images. Rescored on our clean copies they give 35.532 / 33.337 / 30.450, against
its published 35.54 / 33.35 / 30.46, so the rerun is valid. Each of the 100 images is then paired with our full-100
adapted output:

| σ | ours | GRL-B (released) | difference | 95 % CI | images better | median | sign-flip p |
|---|---:|---:|---:|---|---:|---:|---:|
| 15 | 35.501 | 35.532 | −0.031 | [−0.072, +0.010] | 52/100 | +0.004 | 0.15 |
| 25 | 33.402 | 33.337 | **+0.065** | [+0.034, +0.096] | 73/100 | +0.060 | 1.0·10⁻⁴ |
| 50 | 30.671 | 30.450 | **+0.220** | [+0.183, +0.257] | 92/100 | +0.192 | < 10⁻⁵ |

**Reading.**

- Both methods are scored by the same code, on the same images and the same clean copies. The protocol effect that
  [`../../measurement_audit`](../../measurement_audit/README.md) measures, including GRL's own cropping, therefore
  cancels. What remains is the method and the noise draw.
- **At σ 50 the margin is +0.22 dB**, well above the +0.05 dB bar, on 92 of 100 images. **At σ 25 it is +0.065 dB**,
  above the bar but narrowly, on 73 of 100. **At σ 15 there is no gain:** the confidence interval covers zero and the
  point estimate is negative.
- This is the first comparison in this project in which the pipeline is ahead of the strongest released supervised
  denoiser without an external prior, on identical images, with a paired test and a validated reference.
- **The result is not driven by the 12 tuning images.** dev-12 chose the tile size and the step count, so the honest
  subset is the other 88. Excluding dev-12:

  | σ | full 100 | confirm-88 only (no tuning influence) | dev-12 only |
  |---|---|---|---|
  | 25 | +0.065 (73/100, p = 1·10⁻⁴) | **+0.059 (63/88, p = 8.6·10⁻⁴)** | +0.104 (10/12) |
  | 50 | +0.220 (92/100, p < 10⁻⁵) | **+0.209 (80/88, p < 10⁻⁵)** | +0.301 (12/12) |

  The dev images gain more, as expected, but removing them costs only 0.006 and 0.011 dB. The claim rests on images whose
  settings were frozen in advance.
- **What it does not establish.** PTG-RM has no weights, so it cannot enter this comparison. Its published numbers are
  0.17 dB above ours at σ 25 and 0.03 dB below at σ 50, on its own protocol. And the pipeline does test-time training,
  so the class is "no task-specific training plus test-time adaptation", not "training-free".

## Against the published bars (cross-protocol, for reference only)

| σ | ours (full 100) | best published without an external prior | Δ | PTG-RM (CLIP) | Δ |
|---|---:|---|---:|---:|---:|
| 15 | 35.501 | 35.54 (GRL-B) | **−0.039** | 35.70 | −0.199 |
| 25 | 33.402 | 33.35 (GRL-B) | **+0.052** | 33.57 | −0.168 |
| 50 | 30.671 | 30.60 (IPT-V2); 30.59 (Hi-IR) | **+0.071** | 30.64 | **+0.031** |

**Reading.**

- This is the first measured, not projected, full-100 Urban100 result from this pipeline, and it is the first cell where
  it is **above every published method that uses no external prior**: +0.05 at σ 25, +0.07 at σ 50.
- At σ 50 it is also above PTG-RM by +0.031 dB. That is **below the +0.05 dB bar this project set for a SOTA claim**
  and, more importantly, it compares across protocols: PTG-RM's number is on its own copy and scoring rule.
- At σ 15 it is 0.04 dB below GRL-B. The adaptation was never tuned at σ 15, so the pipeline is not uniformly better.
- **Protocol caveat that decides everything.** The measurement audit
  ([`../../measurement_audit`](../../measurement_audit/README.md)) finds that protocol and test-copy choices move a
  fixed denoiser by 0.04–0.16 dB, and that a published 0.08 dB gap *reverses sign* when both methods are scored on the
  same pixels. A +0.05 dB margin is therefore **not resolvable** without rerunning GRL-B / IPT-V2 / Hi-IR on these exact
  draws. That is the next run.
- **Class.** This pipeline does test-time training, so the honest class is "no task-specific training plus certified
  test-time adaptation", not "training-free".
