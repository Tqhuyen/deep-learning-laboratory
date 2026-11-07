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
- **Scope:** Urban100 confirm-88. σ 50 on two further draws (`urban100-seed-1`, `-2`); σ 25 on one further draw
  (`urban100-s25-seed-1`). A second σ 25 draw could not be launched: Kaggle refused the push with "maximum batch GPU
  session count of 2 reached", and the two draws already available agree to 0.003 dB.
- **Draw construction:** the noise seed becomes `1_000_000·(index + 1 + 10·k) + 1000σ + idx` for k = 1, 2, which is
  disjoint from the published set (k = 0).

## Results, σ 25 (two draws)

| draw | unadapted | adapted | gain | per-image SD | images better | worst image |
|---|---:|---:|---:|---:|---:|---:|
| 0 (published set) | 33.1631 | 33.3103 | **+0.1472** | 0.131 | 88/88 | +0.009 |
| 1 | 33.1624 | 33.3121 | **+0.1498** | 0.133 | 88/88 | +0.010 |

- **Across draws:** mean +0.1485 dB; the two per-draw estimates differ by 0.0026 dB.
- **Paired by image:** the per-image gain differs between draws by +0.0026 ± 0.0156 dB, and its ranking is nearly
  identical (Spearman **+0.989**), which is the image-property result of
  [`../../adaptation`](../../adaptation/README.md) seen from another angle.
- **Full-100 at σ 25, two-draw mean:** 33.403 dB, against GRL-B's released 33.337 — **+0.066 dB**.
- **No image harmed** on either draw.

## Results, σ 50 (three draws)

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

**Does not:** PTG-RM still cannot be rerun, so the one comparator that leads the published tables stays outside every
paired test here; and the class of the method is still "no task-specific training plus test-time adaptation".
