# Urban100 (colour): the one-step diffusion posterior mean on large images

Two Kaggle runs (private, 2× Tesla T4):

* [`trnquanghuyn/urban100-one-step`](https://www.kaggle.com/code/trnquanghuyn/urban100-one-step) (32 min) runs
  [`notebooks/urban100_one_step.ipynb`](../../notebooks/urban100_one_step.ipynb). It covers Restormer on identical noise,
  and its whole-image ADM runs were partly out of memory (see below). Scores: [`round1_scores_with_restormer.csv`](round1_scores_with_restormer.csv).
* [`trnquanghuyn/urban100-tiles`](https://www.kaggle.com/code/trnquanghuyn/urban100-tiles) (49 min) runs
  [`notebooks/urban100_tiles.ipynb`](../../notebooks/urban100_tiles.ipynb). It compares the whole image with tiles on
  all 100 images. Scores: [`scores.csv`](scores.csv), [`paired.csv`](paired.csv), [`image_info.csv`](image_info.csv).

**Bottom line.**
- Running ADM on 512-px tiles instead of the whole ~1024-px image raises PSNR by +0.05 / +0.09 / +0.15 dB at
  σ = 15 / 25 / 50. This held on the 88 confirmation images, every image at σ 25 and σ 50, and is significant after
  Holm correction.
- With the flip average this gives 35.345 / 33.227 / 30.468 dB. That beats DMID's released Urban100 outputs on all 100
  images at every σ, so it is the best training-free Urban100 result we know of.
- Supervised GRL-B / Hi-IR (35.54 / 33.35 / 30.59) and PTG-RM (35.70 / 33.57 / 30.64) remain ahead.

## Protocol

- **Noise:** DMID's protocol (sample std exactly σ/255, unclipped). The seeds are those of
  `diffusion-tweedie-benchmark`, where Urban100 is the fourth set.
- **Split, fixed before any tile score:**
  - dev-12, images 8, 16, …, 96, is used for tuning only;
  - confirm-88 is the other 88 images; the pre-registered test is run on these.
- **Estimator:** the one-step ADM posterior mean at the continuous step matched to σ. ×2 averages it with the
  horizontal flip.
- **Tiles:** overlapping tiles blended with linear ramps; 512 px with overlap 128 unless stated.
- **Infrastructure fix.** Round 1 passed 3-D q, k, v to `scaled_dot_product_attention`. The fused kernels need 4-D
  inputs, so it fell back to the math kernel, which builds the N² attention matrix (11 GiB at 1024 px). That ran out of
  memory on 153 of 900 method runs. Round 2 uses 4-D inputs; its output differs from ADM's legacy attention by at most
  4·10⁻³.
  - The earlier ≤ 512-px studies were numerically unaffected, because the math kernel computes the same result.

## Results

Mean PSNR in dB:

| method | evaluations | σ 15 | σ 25 | σ 50 |
|---|---:|---:|---:|---:|
| Restormer, σ-specific, our rerun | 1 | 35.133 | 32.954 | 30.018 |
| DMID, released outputs (one step, own noise draws) | 1 | 35.261 | 33.108 | 30.284 |
| ADM, whole image ×1 | 1 | 35.263 | 33.108 | 30.281 |
| ADM, whole image ×2 | 2 | 35.289 | 33.135 | 30.314 |
| ADM, 512-px tiles ×1 | ≈ 6 | 35.311 | 33.186 | 30.415 |
| **ADM, 512-px tiles ×2** | ≈ 12 | **35.345** | **33.227** | **30.468** |
| GRL-B / Hi-IR, published (supervised) | — | 35.54 | 33.35 | 30.59 |
| PTG-RM, published (supervised + CLIP) | — | 35.70 | 33.57 | 30.64 |

- **Evaluation counts:** for tiles, the count is the number of 512² tile evaluations on a typical 1024×768 image.
- **The whole-image ×1 row reproduces DMID's released numbers.** DMID's Urban100 setting is one step at its integer
  step.

Pre-registered test on confirm-88, tiles-512 ×2 − whole ×2 (sign-flip test with 200,000 patterns, Holm-corrected):

| | σ 15 | σ 25 | σ 50 |
|---|---:|---:|---:|
| mean (95 % CI) | +0.053 ± 0.007 | +0.087 ± 0.010 | +0.147 ± 0.015 |
| images better | 85/88 | 88/88 | 88/88 |
| p (Holm) | < 10⁻⁴ | < 10⁻⁴ | < 10⁻⁴ |

**Other paired comparisons on all 100 images (dB, images better):**

| comparison | noise draws | σ 15 | σ 25 | σ 50 |
|---|---|---:|---:|---:|
| tiles-512 ×2 − Restormer | identical | +0.212 (97) | +0.273 (100) | +0.451 (100) |
| tiles-512 ×2 − DMID released | different | +0.084 (100) | +0.119 (100) | +0.185 (100) |

**Tile size on dev-12 (×1, 12 images; used only to choose the setting for later runs):**

| | σ 15 | σ 25 | σ 50 |
|---|---:|---:|---:|
| whole image | 35.843 | 33.773 | 30.882 |
| attention logits × sqrt(log N / log n_train) | 35.891 | 33.847 | 31.003 |
| 768-px tiles | 35.863 | 33.805 | 30.943 |
| 512-px tiles | 35.913 | 33.879 | 31.062 |
| 384-px tiles | **35.940** | **33.922** | **31.130** |

## Reading

* **Restricting the context fixes most of the Urban100 deficit relative to DMID's whole-image step.**
  - A ~1024-px image gives ADM about 12× the attention tokens it was trained on (1,024 at its first attention level).
  - Fewer tokens per pass help monotonically on Urban100, from the whole image down to 384-px tiles.
  - **It is not a hidden ensemble.** On pixels that get a single tile estimate per pass, tiles ×2 and whole ×2 both
    average exactly two estimates. There the gain is still 84–88 % of the all-pixel gain: +0.044 / +0.076 / +0.130
    dB, with 83 / 88 / 88 of 88 images better ([`hidden_ensemble/`](hidden_ensemble/),
    [`notebooks/urban100_hidden_ensemble.ipynb`](../../notebooks/urban100_hidden_ensemble.ipynb)).
  - Rescaling the attention logits (Jin et al., NeurIPS 2023) recovers about half of the tile gain. Local GroupNorm
    statistics and the convolutional receptive field, which tiling also changes, are untested.
  - The gain grows with σ, as the posterior mean leans more on the prior.
* **Small images behave differently.** On Kodak24 (768×512), 256-px tiles lost 0.014–0.072 dB to the whole image
  ([`../diffusion_benchmark/README.md`](../diffusion_benchmark/README.md)). There is an optimum context size above the
  training resolution; it was chosen on dev-12 as 384 px for the runs that follow.
* **The deficit is now closed at σ 25 and σ 50, and only there.**
  - Full 100 images, frozen pipeline: 33.402 / 30.671 dB at σ 25 / σ 50 and 35.501 at σ 15.
  - Paired against GRL-B's *released outputs* on the same images, that is **+0.065 dB** (73/100 images, p = 1·10⁻⁴)
    and **+0.220 dB** (92/100, p < 10⁻⁵); at σ 15 it is −0.031 dB with the interval covering zero.
  - The whole accounting is in [`full100/`](full100/README.md).
  - Tiles alone, without adaptation, are still 0.12–0.23 dB behind at σ 15/25; it is the test-time adaptation that
    closes the gap, and only on the two higher noise levels.

## A BM3D Wiener stage on top of ADM: no gain (CPU kernel)

- **Run.** [`trnquanghuyn/urban100-nonlocal-wiener`](https://www.kaggle.com/code/trnquanghuyn/urban100-nonlocal-wiener)
  of [`notebooks/urban100_nonlocal_wiener.ipynb`](../../notebooks/urban100_nonlocal_wiener.ipynb). Scores are in
  [`nonlocal/`](nonlocal/).
- **Method.** BM3D's collaborative Wiener stage does block matching on a pilot, then empirical Wiener shrinkage of the
  noisy image's 3-D group spectra. It is blended as λ·W + (1 − λ)·ADM.
- **Oracle gate, clean image as the pilot (dev-12, σ 25):** +0.677 dB at λ = 0.6. Non-local filtering has large
  headroom given a perfect pilot.
- **With the tiles-512 ×2 ADM output as the pilot:** the λ chosen on dev-12 is 0 at both σ 25 and σ 50. Any weight on
  the Wiener output lowers PSNR, so the pre-registered A3 is killed.
- **Reading.** ADM's own estimate already uses the repetition the Wiener stage could exploit, which matches Reviewer
  2's objection. Gains on Urban100 come instead from per-image adaptation ([`../adaptation`](../adaptation/README.md)).
