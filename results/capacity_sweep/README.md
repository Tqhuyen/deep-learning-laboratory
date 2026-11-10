# Is the harm a noise-fitting tax or a content effect? A parameter-count sweep

[`../loss_comparison`](../loss_comparison/README.md) showed that the same adaptation gains on Urban100 and loses on BSD68
at σ 50 even on a convolutional denoiser with no context deficit. So the context-deficit predictor of
[`../adaptation`](../adaptation/README.md) cannot be the whole mechanism, and two candidates were left: a **noise-fitting
tax** (the Recorrupted-to-Recorrupted target carries the image's own noise, so the fitted part grows with σ and with
parameters relative to pixels) and a **content** effect (Urban100 is the repetitive set).

This experiment separates them by varying **only the number of fitted parameters**.

- **Run.** CPU kernel `capacity-sweep-dncnn` (2 h 15 min) of
  [`notebooks/capacity_sweep_dncnn.ipynb`](../../notebooks/capacity_sweep_dncnn.ipynb). Scores:
  [`capacity_sweep.csv`](capacity_sweep.csv).
- **Setup.** DnCNN-gray blind, R2R loss, 60 steps, 2 × 96² crops, Adam at lr 1e-3 — identical to `sure-vs-r2r-cpu`.
  The fitted set is a fixed random subset of the 1,217 gains, the same subset in every case, at 100 / 50 / 25 / 10 %,
  with the held-out gains' gradients zeroed before each step. 24 BSD68 images and 12 centre 512² Urban100 crops, at
  σ 25 and 50. 72 cases, 288 arm-runs.

## Result

Mean gain over the unadapted baseline, dB, by share of the 1,217 gains fitted:

| set, σ | 100 % | 50 % | 25 % | 10 % | 10 % − 100 % | paired p |
|---|---:|---:|---:|---:|---:|---:|
| BSD68 σ 25 | +0.036 | +0.037 | +0.035 | +0.029 | −0.007 | 0.17 |
| **BSD68 σ 50** | **−0.070** | −0.044 | −0.020 | **−0.006** | **+0.064** | **1.5·10⁻⁵** |
| **Urban100 σ 25** | **+0.262** | +0.226 | +0.190 | +0.141 | **−0.121** | **4.7·10⁻⁴** |
| **Urban100 σ 50** | **+0.183** | +0.160 | +0.130 | +0.100 | **−0.083** | **1.3·10⁻²** |

Share of images made worse than the baseline by more than 0.05 dB:

| set, σ | 100 % | 50 % | 25 % | 10 % |
|---|---:|---:|---:|---:|
| BSD68 σ 25 | 0 % | 0 % | 0 % | 0 % |
| **BSD68 σ 50** | **46 %** | 25 % | 12 % | **8 %** |
| Urban100 σ 25 / 50 | 0 % | 0 % | 0 % | 0 % |

## Both effects are real, and they move in opposite directions

**The tax is there.** On the cell where adaptation harms, removing parameters improves the outcome monotonically: gain
rises from −0.070 to −0.006 dB and the harm rate falls from 46 % to 8 %, with 22 of 24 images better at 10 % than at
100 % (p = 1.5·10⁻⁵). That is a dose-response curve in the parameter count, which is what a fitting tax predicts.

**Capacity also buys headroom.** On the cells where adaptation helps, removing parameters costs gain monotonically:
Urban100 loses 0.121 dB at σ 25 and 0.083 dB at σ 50, with 0 of 12 and 3 of 12 images improved. That is the opposite
sign, and it is not noise.

**On BSD68 at σ 25 nothing happens at all** (−0.007 dB, p = 0.17), because there is no harm to remove and no headroom to
lose.

## The pre-registered criteria, read exactly

| criterion | outcome |
|---|---|
| tax: the 10 % subset is ≥ 0.05 dB better than 100 % on BSD68 σ 50 | **pass**, +0.064 dB |
| tax: the 10 % subset reaches ≥ 0 | **fail, by 0.0058 dB** — the gain is −0.006, i.e. indistinguishable from zero in practice but strictly negative |
| content alone: BSD68 σ 50 stays within ±0.02 dB across the sweep | **fail**, the range is 0.064 dB |
| both directions monotone | **yes**, which the pre-registration said to report as *additive* rather than as a win for either |

## What this settles, and what it explains

The adaptation has **two opposing terms that both scale with capacity**:

- **headroom**, from content the prior cannot capture — largest on repetitive images;
- **tax**, from fitting the image's own noise — largest at high σ and high parameter count.

The net gain is their difference. That single sentence explains every pattern this project has measured:

- Urban100 (repetitive, large, moderate σ): headroom dominates → gains.
- BSD68 / Set12 at σ 50 (small, high noise): tax dominates → losses, and the losses shrink when fewer parameters are
  fitted.
- σ 15 on Urban100: headroom smaller → gains shrink, which is why that cell does not overtake GRL-B.
- Kodak24 / McMaster: neither term is large → roughly zero.

It also resolves an earlier retraction. The first draft proposed a "noisy values per adapted parameter" rule and it was
withdrawn because image size did not predict the gain *across* sets. The rule was not wrong that capacity matters; it was
wrong to read a single cross-set comparison as a capacity law. Headroom and tax move together with capacity, and only an
experiment that holds one fixed — this one — can see both.

## Limits

- One denoiser, one loss, one step budget, and a fixed random subset rather than a nested one. A nested sweep would show
  whether the two terms have the same shape.
- The 10 % arm reaches 122 gains, so its "≈0" is a statement about *that* size, not about the limit of the sweep.
- DnCNN is not the diffusion prior. That the tax appears on both is evidence that it is a property of the loss and the
  data, not of ADM.
