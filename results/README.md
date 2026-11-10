# Results

One directory per study. Each holds a `README.md` that states what was run, what it found, what it did **not**
establish, and points at the CSVs beside it and the notebook source in [`../notebooks/`](../notebooks).

Two conventions run through all of them:

- **Claims are paired or they are not claims.** A margin over a published number is shown only as a reference column,
  because the study in `measurement_audit/` shows the scoring protocol moves a *fixed* denoiser by more than those
  margins. Anything asserted is asserted against a rerun on the same pixels.
- **Failures are reported.** Per-image adaptation loses on three of the five sets it was run on; DMID's own
  multi-evaluation chain is worse than one step; the Wiener stage gains nothing with a real pilot; calibration,
  ensembles and fine-tuning each move PSNR by at most 0.02 dB. Those are results, and they are in the reports.

## The denoising study

The group of directories that make one paper. Start at [`paper/README.md`](paper/README.md), which indexes the draft,
the claims ledger and the prior-art table.

| study | what it asks | headline |
|---|---|---|
| [`diffusion_benchmark/`](diffusion_benchmark/README.md) | how much of DMID's result does one evaluation give, and what do the remaining knobs add? | within 0.02 dB; every knob ≤ 0.02 dB; DMID's own chain is 0.11–0.14 dB *worse* than one step |
| [`gray_lift/`](gray_lift/README.md) | can an RGB prior denoise grayscale? | yes, through an exact lift; the first diffusion-prior Set12/BSD68 numbers, beating every released supervised gray model on identical draws |
| [`urban100/`](urban100/README.md) | why is the prior weak on large images, and does tiling fix it? | it is context, not averaging (84–88 % of the gain survives single-coverage pixels) |
| [`urban100/full100/`](urban100/full100/README.md) | what does the frozen pipeline reach on all 100 images? | 33.402 / 30.671 dB at σ 25 / 50, **+0.065 / +0.220 dB** over GRL-B's released outputs, paired |
| [`urban100/seed_replication/`](urban100/seed_replication/README.md) | do those margins survive fresh noise draws? | yes: three draws at σ 50, two at σ 25, 88/88 images better every time |
| [`adaptation/`](adaptation/README.md) | when does per-image adaptation help? | it is a property of the image (ρ = 0.98 across draws) predicted by one measured deficit (ρ = −0.44…−0.56) |
| [`capacity_sweep/`](capacity_sweep/README.md) | is the harm a fitting tax or a content effect? | both: varying only the fitted parameter count moves harm and headroom in opposite directions |
| [`loss_comparison/`](loss_comparison/README.md) | is our loss the weak link against GainTuning's SURE, at equal budget? | R2R beats SURE in all four cells (+0.07…+0.24 dB) on DnCNN; the harm pattern transfers across architectures |
| [`measurement_audit/`](measurement_audit/README.md) | how much can a published number move without the method changing? | cropping 0.04–0.17 dB, colour channel 1.6–2.8 dB, clipped input up to 3.8 dB; two architectures agree to 0.02 dB |
| [`real_noise/`](real_noise/README.md) | does any of it transfer to camera noise? | parity with DMID-d only, below the newer bars; reported as analysis, not as a claim |
| [`adm_finetune/`](adm_finetune/README.md), [`prior_ensembles/`](prior_ensembles/README.md) | do the obvious routes help? | no: ±0.001 dB and ≤ 0.004 dB |

## The rest

- [`recurrent_comparison/`](recurrent_comparison/README.md) — RNN / GRU / LSTM against a transformer at matched
  parameters, with diagnostics: gradient reach, loss landscapes, BPC against evaluation context.
- [`jepa_guided_diffusion/`](jepa_guided_diffusion/README.md) — an earlier attempt at conditioning ADM on a JEPA
  adapter; the null result that motivated the per-image route above.
