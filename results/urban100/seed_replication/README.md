# Replication of the Urban100 σ 50 result on two further noise draws

The headline σ 50 result of [`../full100`](../full100/README.md) rested on one noise draw. This run repeats it on two
fresh draws, which is the standard this project set for itself before any of these runs (three seeds, paired test,
Holm correction).

- **Runs.** GPU kernels [`urban100-seed-1`](https://www.kaggle.com/code/trnquanghuyn/urban100-seed-1) (3 h 15 min) and
  [`urban100-seed-2`](https://www.kaggle.com/code/trnquanghuyn/urban100-seed-2) (3 h 28 min) of
  [`notebooks/urban100_seed_replication_1.ipynb`](../../../notebooks/urban100_seed_replication_1.ipynb) and
  `…_2.ipynb`.
- **Scores:** [`seed1_scores.csv`](seed1_scores.csv), [`seed2_scores.csv`](seed2_scores.csv), and their `*_tests.csv`.
- **Settings:** identical to the frozen pipeline — 384-px tiles, ×2 flip, 90 R2R steps, Muon/AdamW branch, α = 0.5. The
  only change is the noise draw and the adaptation's own random seed.
- **Scope:** Urban100 confirm-88 at σ 50 only. σ 25 was not replicated (GPU budget); claims at σ 25 remain
  single-draw.
- **Draw construction:** the noise seed becomes `1_000_000·(index + 1 + 10·k) + 1000σ + idx` for k = 1, 2, which is
  disjoint from the published set (k = 0).

## Results

confirm-88, σ 50, ×2 flip:

| draw | unadapted | adapted | gain | per-image SD | images better | images worse by > 0.05 dB |
|---|---:|---:|---:|---:|---:|---:|
| 0 (published set) | 30.4317 | 30.5862 | **+0.1545** | 0.142 | 88/88 | 0 |
| 1 | 30.4344 | 30.5835 | **+0.1491** | 0.141 | 88/88 | 0 |
| 2 | 30.4349 | 30.5864 | **+0.1515** | 0.145 | 87/88 | 0 |

- **Across draws:** mean gain **+0.1517 dB**, standard deviation **0.0027 dB**.
- **Pooled over 264 image–draw pairs:** mean +0.1517 dB, 263/264 images better, worst −0.006 dB, sign-flip
  p < 10⁻⁵.
- **Full-100 at σ 50, using the three-draw mean:** 30.670 dB, unchanged from the single-draw figure. Against GRL-B's
  released outputs (30.450) that is **+0.220 dB**; against Hi-IR (30.59) +0.080; against PTG-RM (30.64) +0.030.

## By-product: the real size of the noise-draw effect

The unadapted estimate is a deterministic function of the noisy image, so its set mean over 88 images moves only with
the draw:

| quantity | SD across the three draws |
|---|---:|
| unadapted set mean | **0.0017 dB** |
| adapted set mean | 0.0016 dB |

An 88-image set mean therefore has a draw-to-draw spread of about **0.002 dB**, not the ±0.02 dB an earlier draft of this
project assumed. Two consequences:

1. **Cross-draw comparisons are nearly exact at the set level.** A comparison between our mean and a number computed on
   another draw carries ~0.002 dB of draw uncertainty, which is two orders of magnitude below the margins being
   discussed. Per-image pairing adds its own variance, which the paired tests already report.
2. **The earlier "±0.02 dB draw variance" sentence was wrong and has been removed** from the paper. The deficit of one
   step against DMID-d (0.014 / 0.031 dB at σ 25 / 50) is therefore real, not draw noise.

## What this does and does not establish

**Does:** the σ 50 gain is not a draw artefact. Across two independent draws it reproduces to within 0.005 dB of the
original, on 263 of 264 image–draw pairs.

**Does not:** the σ 25 gain is still single-draw; PTG-RM still cannot be rerun; and the class of the method is still
"no task-specific training plus test-time adaptation".
