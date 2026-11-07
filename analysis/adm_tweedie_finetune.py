"""Paired statistics for ADM fine-tuned on the denoising steps.

The Kaggle notebook ``notebooks/adm_tweedie_finetune.ipynb`` (kernels ``trnquanghuyn/adm-tweedie-finetune``, the pilot,
and ``trnquanghuyn/adm-tweedie-finetune-long``) fine-tunes ADM 256x256 uncond with its ε loss on the denoising steps
and scores the base and fine-tuned one-step estimates on CBSD68, Kodak24, McMaster and Urban100 with the seeds of
``diffusion-tweedie-benchmark``. This script joins those scores with DMID's released outputs and the supervised
baselines of ``diffusion-tweedie-benchmark`` (identical noisy images on the first three sets) and writes:

* ``means.csv``: mean PSNR/SSIM/LPIPS per set, σ and method, plus the best published PSNR per cell;
* ``paired_tests.csv``: the contrasts below, per set and σ and pooled per σ (as in diffusion_tweedie_benchmark);
* ``training.png``: validation PSNR (DIV2K valid) against training step.

    python analysis/adm_tweedie_finetune.py --run-output kaggle_output/sota/runC/outputs \\
        --benchmark-output kaggle_output/sota/runA/outputs --output-directory results/adm_finetune
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from diffusion_tweedie_benchmark import DMID, METRICS, SIGMAS, paired_tests


CONTRASTS = {
    "fine-tuned ×1 − base ×1": ("fine-tuned ×1", "base ×1"),
    "fine-tuned ×2 − fine-tuned ×1": ("fine-tuned ×2", "fine-tuned ×1"),
    "fine-tuned ×1 − DMID-d": ("fine-tuned ×1", DMID),
    "fine-tuned ×2 − DMID-d": ("fine-tuned ×2", DMID),
    "fine-tuned ×2 − Restormer": ("fine-tuned ×2", "Restormer"),
    "fine-tuned ×2 − SwinIR": ("fine-tuned ×2", "SwinIR"),
    "base ×1 − DMID-d": ("base ×1", DMID),
    "fine-tuned two steps (η=0) − fine-tuned ×1": ("fine-tuned two steps (η=0)", "fine-tuned ×1"),
    "fine-tuned blind ×1 − Restormer blind": ("fine-tuned blind ×1", "Restormer blind"),
    "fine-tuned blind ×1 − fine-tuned ×1": ("fine-tuned blind ×1", "fine-tuned ×1"),
}
BASELINES = ("Restormer", "SwinIR", "SCUNet", "DRUNet", "Restormer blind")
BEST_PUBLISHED = {  # PSNR at σ = 15, 25, 50: (value, source)
    "CBSD68": ((34.48, "MambaIR/v2"), (31.86, "DMID-d"), (28.72, "DMID-d")),
    "Kodak24": ((35.51, "DMID-d"), (33.12, "DMID-d"), (30.14, "DMID-d")),
    "McMaster": ((35.73, "MambaIRv2"), (33.49, "DMID-d"), (30.50, "DMID-d")),
    "Urban100": ((35.46, "Hi-IR"), (33.34, "Hi-IR"), (30.59, "Hi-IR")),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-output", type=Path, required=True, help="outputs directory of adm-tweedie-finetune(-long)")
    parser.add_argument("--benchmark-output", type=Path, required=True, help="outputs directory of diffusion-tweedie-benchmark")
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def load(run: Path, benchmark: Path) -> pd.DataFrame:
    ft = pd.read_csv(run / "scores.csv")
    dmid = pd.read_csv(benchmark / "dmid_released_scores.csv")
    base = pd.read_csv(benchmark / "scores.csv")
    images = set(zip(ft.dataset, ft.image))
    keep = lambda df: df[[(d, i) in images for d, i in zip(df.dataset, df.image)]]
    return pd.concat([ft, keep(dmid), keep(base[base.method.isin(BASELINES)])], ignore_index=True)


def means(scores: pd.DataFrame) -> pd.DataFrame:
    table = scores.groupby(["dataset", "sigma", "method"])[list(METRICS)].mean().reset_index()
    best = [dict(dataset=ds, sigma=s, method=f"best published ({src})", psnr=v)
            for ds, cells in BEST_PUBLISHED.items() if ds in set(table.dataset) for s, (v, src) in zip(SIGMAS, cells)]
    return pd.concat([table, pd.DataFrame(best)], ignore_index=True)


def plot_training(log: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(log)
    figure, axis = plt.subplots(figsize=(5.5, 3.6))
    for s in SIGMAS:
        column = f"val_psnr_{s}"
        if column in frame:
            axis.plot(frame.step, frame[column] - frame[column].iloc[0], "o-", label=f"σ = {s}")
    axis.set(xlabel="training step", ylabel="Δ validation PSNR vs step 0, dB", title="DIV2K valid 0801–0810, one step (EMA)")
    axis.axhline(0, color="k", lw=0.6)
    axis.legend()
    plt.tight_layout()
    plt.savefig(out / "training.png", dpi=120)
    plt.close(figure)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    args = parse_arguments()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    scores = load(args.run_output, args.benchmark_output)
    table = means(scores)
    table.to_csv(args.output_directory / "means.csv", index=False, float_format="%.6g")
    tests = paired_tests(scores, CONTRASTS)
    tests.to_csv(args.output_directory / "paired_tests.csv", index=False, float_format="%.6g")
    plot_training(json.loads((args.run_output / "train_log.json").read_text(encoding="utf-8")), args.output_directory)
    print(table.pivot_table(index="method", columns=["dataset", "sigma"], values="psnr").round(3).to_string())
    psnr = tests[tests.metric == "psnr"]
    for _, r in psnr.iterrows():
        print(f"{r.contrast:32s} {r.dataset:9s} σ{r.sigma:<3d} {r.mean_difference:+.3f} dB [{r.ci_low:+.3f}, {r.ci_high:+.3f}] "
              f"better {r.first_better}/{r.n}  Holm p={r.p_holm:.3g}")
    print(f"wrote {args.output_directory}")


if __name__ == "__main__":
    main()
