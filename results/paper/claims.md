# Claims ledger

Every claim the paper is allowed to make, with its evidence, its statistic, and what would falsify it. Anything not in
this table is not claimed. Built after the Phase-1 deconstruction, so each row has already survived one adversarial
pass.

Legend for **status**: `supported` means the evidence exists in this repository and has been re-checked from the CSVs;
`single-draw` means it rests on one noise draw; `negative` means we report a failure.

## Claimed

| # | claim | evidence | statistic | what would falsify it | status |
|---|---|---|---|---|---|
| **C1** | On Urban100 colour at σ = 25, the pipeline (384-px tiles ×2 + 90-step R2R) beats GRL-B's released outputs on the same 100 images | [`../urban100/full100`](../urban100/full100/README.md), `scores.csv`, [`../measurement_audit/grl_rescored.csv`](../measurement_audit/grl_rescored.csv) | **+0.065 dB**, 95 % CI [+0.034, +0.096], 73/100 images, sign-flip p = 1·10⁻⁴. On the 88 images whose settings were not tuned on them: **+0.059, 63/88, p = 8.6·10⁻⁴**. Replicated on a **second draw**: +0.1472 / +0.1498 dB on the 88, 88/88 images better on both, per-image ranking ρ = 0.989 | GRL-B rerun on our exact draws gives ≤ 0; or any released model without an external prior scores higher on these pixels | supported, **2 draws** |
| **C2** | The same at σ = 50 | [`../urban100/seed_replication`](../urban100/seed_replication/README.md) | **+0.220 dB** against GRL-B on the full 100, from the **three-draw** mean. Confirm-88 gain is **+0.1517 ± 0.0027 dB** across three independent draws; pooled 264 image–draw pairs, 263/264 better, p < 10⁻⁵. Holdout-only, full 100: +0.209 | as C1 | **supported, 3 draws** |
| **C3** | At σ = 15 the pipeline has **no** advantage | as C1 | −0.031 dB, CI [−0.072, +0.010], 52/100, p = 0.15 | a gain appears under a σ15-specific setting tuned on data disjoint from Urban100 | **negative, reported as such** |
| **C4** | The gray lift beats every released supervised gray denoiser on **identical draws**, except where our reference reruns fail | [`../gray_lift`](../gray_lift/README.md) | BSD68 vs Xformer **68/68** images at every σ (+0.057 / +0.078 / +0.106); vs Restormer **612/612 pairs** (3 seeds × 68 images × 3 σ), mean **+0.103 dB**, and Set12 **107/108**, mean +0.151; Set12 vs GRL-B σ 50 **+0.192 dB**, 11/12, p = 9.3·10⁻⁴ | any paired comparison reversing on a validated rerun | supported |
| **C5** | No diffusion-prior method had reported gray Set12 / BSD68 Gaussian numbers before this | [`prior_art.md`](prior_art.md) | search of 2021–2026 diffusion-restoration papers | any paper reporting them | supported (absence of evidence, stated as such) |
| **C6** | One step plus a flip is within 0.02 dB of DMID-d's 1500 evaluations on the three small colour sets | [`../diffusion_benchmark`](../diffusion_benchmark/README.md), `paired_tests.csv` | σ15 +0.001 (n.s.), σ25 −0.014, σ50 −0.031 | a paired rerun of DMID-d on our draws showing a larger gap | supported |
| **C7** | DMID's own three-evaluation chain is **worse** than one step on matched Gaussian noise | [`../diffusion_benchmark/tile_rule`](../diffusion_benchmark/tile_rule/README.md) | −0.128 / −0.137 / −0.114 dB, **0/110 images** better at every σ | any set or σ where the chain wins | supported |
| **C8** | The tile gain is not overlap averaging | [`../urban100/hidden_ensemble`](../urban100/hidden_ensemble/README.md) | 84–88 % of the gain survives on pixels with one estimate per pass; 88/88 images at σ 25/50 | a GroupNorm-localisation arm recovering most of the remaining gain (then the mechanism would be statistics, not context) | supported for averaging; mechanism open |
| **C9** | Per-image adaptation is image-specific, not a generic large-image correction | [`../adaptation`](../adaptation/README.md), `gate4/gains.csv` | parameters transferred to another image lose **−0.10 to −0.11 dB** in all three modes, and clean-target **oracle** parameters also transfer negatively (−0.10) | a pooled adaptation retaining ≥ 50 % of the per-image gain | supported |
| **C10** | The scoring protocol moves a **fixed** denoiser by more than the margins being ranked, and the effect belongs to the protocol rather than to the model | [`../measurement_audit`](../measurement_audit/README.md) | crop / border **0.04–0.17 dB**; RGB vs Y **1.6–2.8 dB**, shrinking with σ; clipped-quantised input **−0.48 / −0.99 / −2.46 dB** (mean over three sets) at σ = 15 / 25 / 50, up to **−3.81 dB** on McMaster, growing monotonically with σ. Two architecturally different denoisers agree on the crop shift to **0.012 dB** over **9** cells and on the colour-space shift to **0.024 dB**. On Kodak24 σ 15/25 the published ordering of two methods **reverses** on identical pixels. For scale, the noise draw moves an 88-image set mean by **0.0017 dB** | a third denoiser, or a set, where the same protocol shift differs by more than 0.05 dB between models | supported, two denoisers |
| **C11** | The dataset copies we use are correct | C4, the gray README's reproduction table, and the measurement audit | Restormer and Xformer reproduce published BSD68 numbers within **0.005 dB** at every σ, Set12 within 0.004, and Restormer reproduces Urban100 σ 15 at **35.133 against 35.13** published | a copy-level pixel difference found between our test sets and the publishers' | supported |
| **C12** | Whether per-image adaptation helps is a reproducible property of the **image**, and it is predicted by the prior's context deficit measured before adapting | [`../adaptation`](../adaptation/README.md) §"What predicts whether adaptation helps" | per-image gain across three noise draws: Spearman **+0.98** pairwise. Context deficit vs gain, at every σ on 100 images: ρ = **−0.441 / −0.517 / −0.558** at σ = 15 / 25 / 50 (p ≤ 4·10⁻⁶); partial ρ = −0.54 at σ 50 after controlling for baseline PSNR (p = 6·10⁻⁸), while baseline PSNR itself gives ρ = +0.04 (p = 0.74). **Across σ the two rank identically (rank correlation −1.0), and across four set/σ cells likewise.** The σ 15 deficit is −0.082 dB against −0.187 at σ 50, which is why σ 15 is the one cell without a win | a set with a large context deficit on which adaptation loses, or a set with a deficit below 0.03 dB on which it gains | supported within Urban100 |
| **C13** | The adaptation gain is not a noise artefact | [`../urban100/seed_replication`](../urban100/seed_replication/README.md) | three draws, gain +0.1545 / +0.1491 / +0.1515, SD 0.0027; per-image ranking stable at ρ = 0.98 | per-image gains disordering across draws | supported |

## Explicitly not claimed

| not claimed | why |
|---|---|
| State of the art on CBSD68 / Kodak24 / McMaster | One step ×2 is within 0.02 dB of DMID-d but below it at σ 50 on all three sets; PTG-RM is 0.07–0.26 dB higher. |
| State of the art on Urban100 at σ 15 | C3: no advantage. |
| Beating PTG-RM | It releases no denoising weights, so it cannot be rerun; its published margin is 0.17 dB at σ 25 and −0.03 dB at σ 50. Cross-protocol, therefore not claimed. |
| "Training-free" | The pipeline runs 90 optimiser steps per test image. The class is "no task-specific training plus test-time adaptation". |
| Real-noise parity or SOTA | Parity with DMID-d only, and below Cheng & Tan (38.01 / 38.71) and P2N (34.00). |
| Paired wins over SCUNet / DRUNet on BSD68 | Their reruns land 0.10–0.35 dB below their published BSD68 numbers while matching on Set12, so the reruns are not a valid reference (C11 explains why our copy is fine). |
| Tile rule R as a contribution | +0.017 / +0.010 / −0.019 on Kodak24 / McMaster / CBSD68. Within the protocol budget. |
| Any cross-paper margin below ~0.15 dB | C10. |
| The causal mechanism of the adaptation gain | Size, content and passes per pixel are confounded; the "capacity rule" of an earlier draft was retracted. |

## Conditions under which C1 and C2 would be withdrawn

1. ~~The 3-seed replication fails.~~ **Run and passed:** +0.1545 / +0.1491 / +0.1515 dB on draws 0 / 1 / 2, pooled
   263/264 image–draw pairs, p < 10⁻⁵. C2 is a three-draw result. C1 (σ 25) was replicated on one further draw (+0.1472 / +0.1498 dB,
   88/88 images on both).
2. **A released model without an external prior beats us on these pixels.** MaIR / MaIR+ publishes checkpoints. If one
   of them, rerun on our draws, exceeds 33.402 / 30.671, C1 and C2 are void.
3. ~~A copy-level defect in Urban100.~~ **Checked:** Restormer reproduces its published Urban100 σ 15 number, 35.133
   against 35.13. Our copy is the standard one.
