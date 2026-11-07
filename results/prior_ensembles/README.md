# Class-mixture posterior means and ensembles of pretrained diffusion priors

Kaggle run [`trnquanghuyn/diffusion-prior-ensembles`](https://www.kaggle.com/code/trnquanghuyn/diffusion-prior-ensembles)
(private, 2× Tesla T4, 22 min) of
[`notebooks/diffusion_prior_ensembles.ipynb`](../../notebooks/diffusion_prior_ensembles.ipynb). It uses the protocol and
noise seeds of [`../diffusion_benchmark`](../diffusion_benchmark/README.md), so every method sees identical inputs.

**Question.** Flips of one network make correlated errors: the horizontal-flip average adds only ~0.013 dB. Do
*different* pretrained networks, or a class-mixture posterior mean, add more?

**Bottom line.** They do not. All five networks together beat the unconditional model's flip average by
+0.002 / +0.002 / +0.004 dB, costing about 10 evaluations instead of 2. The class-conditional mixture is slightly worse
than the unconditional model, and the 512×512 model is 0.1 dB worse.

## Estimators

* **Embedding:** every network gets the noisy image at the matched step (Tweedie, one evaluation per class or flip).
* **U:** ADM 256×256 unconditional. U ×2 averages it with its horizontal flip.
* **C256 / C512:** ADM class-conditional at 256 or 512, mixed over the top-3 classes:
  $E[x\mid y] = \sum_c p(c\mid y)E[x\mid y,c]$. Here $p(c \mid x_t)$ comes from OpenAI's noise-aware classifier at the
  same step, averaged over five crops.
* **FT:** the unconditional model fine-tuned on DIV2K ([`../adm_finetune`](../adm_finetune/README.md)).
* **Ensembles:** equal weights, nothing tuned.
* **Numerics:** the 512 model runs in fp16 (checked finite).

## Results (CBSD68 and McMaster, 86 images per σ)

All 72 Kodak24 jobs ran out of GPU memory: a batch of three class-conditional passes at 768×512 is too much on a T4.
The PSNR table is [`summary.json`](summary.json). Paired against U ×2 (sign-flip test):

| PSNR − U ×2, dB | σ 15 | σ 25 | σ 50 |
|---|---:|---:|---:|
| C256 class mixture | −0.012 (4/86 better) | −0.016 | −0.023 |
| C512 class mixture | −0.094 (1/86) | −0.101 | −0.106 |
| FT (one evaluation) | −0.016 | −0.017 | −0.020 |
| U + C256 | +0.001 | −0.002 | −0.005 |
| U + C256 + C512 | −0.005 | −0.006 | −0.005 |
| U ×2 + C256 + C512 + FT | +0.002 (55/86) | +0.002 (52/86) | +0.004 (53/86) |
| DMID-d, released (1500 evaluations) | −0.011 | +0.001 | +0.018 |

* **The classifier is unsure on these images.** Its mean top-1 probability is 0.11 at 256 and 0.18 at 512
  ([`classes.json`](classes.json)). CBSD68 and McMaster are not ImageNet-style object photos, so the class mixture
  behaves like a slightly worse unconditional model.
* **The 512×512 model is consistently worse,** by 0.1 dB at native resolution.
* **Different networks behave like flips.** Their errors are as correlated as those of one network's flips, so
  equal-weight ensembles land within ±0.006 dB of U ×2.

## Reading

This is the third route to stall at the same level, after training-free changes
([`../diffusion_benchmark`](../diffusion_benchmark/README.md)) and fine-tuning on DIV2K
([`../adm_finetune`](../adm_finetune/README.md)). Every route lands within ±0.02 dB of DMID-d.
