# Prior art and the gap this paper fills

Compiled during the Phase-1 deconstruction and the Phase-2 literature scan. Every entry was checked against the paper
or repository it cites. This file backs claims **C5** and the related-work positioning in
[`draft.md`](draft.md).

## Every component of the method has prior art

| component | prior art | what that work established | what it did not do |
|---|---|---|---|
| one-step posterior mean at the matched step | Kawar, Vaksman & Elad, ICCVW 2021 ([arXiv 2101.09552](https://arxiv.org/abs/2101.09552)); DMID §5.4.2 ([arXiv 2307.03992](https://arxiv.org/html/2307.03992v5)) | the score network at the matched step is an MMSE denoiser; DMID reports that one step is optimal without repetitions | measure the paired gap against DMID-d's 1500-evaluation ensemble, or the chain's loss on Gaussian noise |
| gray lift as a spectral top-up | DDRM, Kawar et al. 2022, Eq. 4–5 ([arXiv 2201.11793](https://arxiv.org/html/2201.11793)), App. D.4 | each spectral component gets variance σ_T² − σ_y²/s_i²; colourisation uses H = channel mean, s = 1/√3 | report any grayscale denoising benchmark |
| attention rescaling for unseen resolution | Jin et al., NeurIPS 2023 ([proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/file/e0378e0c642b1d292fcb224e8d5a39b3-Paper-Conference.pdf)) | logit scale √(log N / log n_train) | apply it to denoising; it recovers about half our tile gain |
| a 256-px prior used on larger images | Cheng & Tan, NeurIPS 2024 ([arXiv 2410.17521](https://arxiv.org/html/2410.17521v1)) §3.4, Table 6 | ADM-256 as a "local" prior beats ADM-512 (CC 38.01 vs 37.01) | quantify the tile-size trade-off, or separate context from averaging |
| overlapping patch inference with diffusion priors | WeatherDiffusion ([arXiv 2207.14626](https://arxiv.org/abs/2207.14626)); StableSR ([arXiv 2305.07015](https://arxiv.org/abs/2305.07015)); Hu et al. ([arXiv 2410.11730](https://arxiv.org/abs/2410.11730)) | patch-wise priors work at arbitrary size and beat whole-image models | show the gain is not overlap averaging |
| per-image gain adaptation | GainTuning, Mohan et al., NeurIPS 2021 ([arXiv 2107.12815](https://arxiv.org/html/2107.12815v1)) | per-channel gains tuned on one test image; gains largest on repetitive content; "does not require early stopping" with a SURE loss | apply it to a diffusion prior; it is the missing baseline for our Section 4.3 |
| single-image test-time adaptation of denoisers | LAN (CVPR 2024), TTAD (ICCV 2025), SCD, DDIP | adaptation of small denoisers, 10–1200 iterations | report per-image harm rates, or gate the adaptation |
| SURE without ground truth | Ramani et al., TIP 2008; UNSURE (ICLR 2025); SURE-conformal (2025) | unbiased MSE estimates; marginal coverage | an adapt-or-not decision for a *large* prior |
| data scale, not architecture | Klug & Heckel, ICLR 2023 ([arXiv 2209.13435](https://arxiv.org/pdf/2209.13435)), Table 3 | SwinIR at 100 k ImageNet images: 31.84 / 33.14 / 33.44 / 33.21 at σ 25; scaling exponent flattens to N^0.0019 | compare against a diffusion prior on identical draws |
| diffusion priors are not a special object | Kadkhodaie et al., ICLR 2024 ([arXiv 2310.02557](https://arxiv.org/abs/2310.02557)) | networks trained on disjoint data learn nearly the same score function once the data is large enough | draw the consequence for denoising benchmarks |

## The gap this paper fills

**No diffusion-prior paper reports grayscale Set12 or BSD68 Gaussian denoising.** The training-free classics do: BM3D
31.07 / 28.57 / 25.62 and WNNM 31.37 / 28.83 / 25.87 on BSD68, as tabulated in the DnCNN paper. Every diffusion
restoration paper checked (DMID, LCDD, DIIP, PnP-Flow, PaDIS, DDRM, DDNM, DPS, DiffPIR, GDP) reports colour only,
256² crops, or non-standard noise. This is an absence-of-evidence claim and is stated as such.

**No paper measures what the scoring protocol costs against the margins being ranked.** The nearest neighbours are:

| work | what it varied | what it did not |
|---|---|---|
| NTIRE 2025 / 2026 σ 50 reports | teams, one hidden draw | no confidence intervals; top four within 0.03 dB still ranked |
| Eulig et al., Medical Physics 2024 | training seeds, CT | fixed noise, unpaired tests, not natural images |
| Christodoulou et al., MICCAI 2024 | reconstructed CIs from 221 papers | segmentation; CIs imputed, not rerun |
| Ramos et al., ICCVW Findings 2025 | ImageNet/LAION duplicates | no restoration test sets |
| InverseBench (ICLR 2025), SSIBench (MELBA 2026) | solver benchmarks | exclude natural-image denoising |
| Su et al. (arXiv 2503.13074) | ground-truth quality in SR | not protocol or copy variance |

## The one comparison that is not possible

**PTG-RM** (CVPR 2024, [arXiv 2403.06793](https://arxiv.org/html/2403.06793v2)) holds the best published numbers in
most Gaussian cells. Its repository releases only a SID low-light checkpoint
([repo](https://github.com/xiaogang00/pretrain_model_boost_restoration)), so it cannot be rerun and cannot enter a
paired test. The same holds for Hi-IR and IPT-V2. **MaIR / MaIR+ does publish checkpoints**
([drive](https://drive.google.com/drive/folders/1YYmIVTyynLg-Kfu-mviq24WdVkJu-S3M)), which makes it the best
substitute reference for a second released comparator.
