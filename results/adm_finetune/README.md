# Pilot: fine-tuning ADM on the denoising steps

Kaggle run [`trnquanghuyn/adm-tweedie-finetune`](https://www.kaggle.com/code/trnquanghuyn/adm-tweedie-finetune)
(private, 2× Tesla T4, 1 h 59 min). It is version 1 of
[`notebooks/adm_tweedie_finetune.ipynb`](../../notebooks/adm_tweedie_finetune.ipynb), whose knobs now default to a longer
run; the pilot's values are noted next to them. Statistics come from
[`analysis/adm_tweedie_finetune.py`](../../analysis/adm_tweedie_finetune.py).

**Bottom line.** Fine-tuning moved the one-step estimate by +0.001 dB on average (pooled over 114–117 images per σ,
not significant). Training-free changes had already stalled at DMID-d's level
([`../diffusion_benchmark`](../diffusion_benchmark/README.md)), and this recipe does not lift that ceiling either.

## Setup

* **Training input and loss.** For $y = x + \sigma n$, the scaled noisy image $\sqrt{\bar\alpha}(2y-1)$ with
  $\bar\alpha = 1/(1+4\sigma^2)$ is exactly a forward-diffusion state with $\varepsilon = n$. ADM's own ε loss on such
  states therefore trains the one-step estimate directly.
* **Training data:** σ ~ U[5, 60]/255 on DIV2K train (800 images). Random crops are taken at a random scale in [¼, 1],
  resized to 256 px, and flipped horizontally.
* **Optimisation:**
  * AdamW at lr 2·10⁻⁵, 100 warm-up steps, then cosine decay to 10 %. The notebook now trains with Muon (AdamW for the
    first and last convolutions, biases and norms); the numbers below are from the AdamW run.
  * fp16 autocast, gradient checkpointing, EMA with a 150-step half-life.
  * DDP on 2 GPUs, effective batch 8.
* **Run:** 1,678 steps in 107 min (3.6 s per step, peak memory 8.8 GiB per GPU).
* **Evaluation:** one step, on the seeds of `diffusion-tweedie-benchmark`, so the base model reproduces that run.

## Result

Validation (DIV2K valid 0801–0810) rose by 0.014 / 0.012 / 0.008 dB at σ = 15 / 25 / 50. Most of that came in the first
400 steps ([`training.png`](training.png)).

Test sets, fine-tuned minus base, one step ([`paired_tests.csv`](paired_tests.csv)):

| PSNR, dB | σ 15 | σ 25 | σ 50 |
|---|---:|---:|---:|
| CBSD68 (68) | −0.008 (10/68 better) | −0.005 | −0.004 |
| Kodak24 (24) | +0.009 (21/24) | +0.010 (23/24) | +0.006 |
| McMaster (18) | +0.004 | +0.004 | +0.003 |
| Urban100 (subset) | +0.019 (7 images) | +0.024 (4) | +0.028 (4) |
| pooled | −0.001 | +0.001 | +0.001 |

Most Urban100 images ran out of GPU memory: 75 of 90 failed at full resolution with two networks loaded.
`scaled_dot_product_attention` was given non-contiguous q, k, v and fell back to the quadratic-memory kernel. Later
notebooks make them contiguous.

## Reading

* **Fine-tuning moves the estimate the way the data's domain predicts.** It nudges toward DIV2K's statistics: slightly
  better on Kodak24 and the few Urban100 images (closer to DIV2K), slightly worse on CBSD68.
* **The estimate itself barely moves.** The validation curve flattens at +0.01 dB, so at these steps ADM's ε-network is
  already close to what this data and recipe can teach it.
* **Not tested:** larger learning rates, longer schedules, more data (Flickr2K, WED) and parameter-efficient variants.

## Files

* `scores.csv`, `summary.json` and `train_log.json` are copied from the run.
* `means.csv`, `paired_tests.csv` and `training.png` are written by the analysis script.
* The fine-tuned EMA weights (`adm_ft_ema.pt`, 1.1 GB) stay in the Kaggle output.
