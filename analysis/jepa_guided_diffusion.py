"""Paired statistics and adapter diagnostics for the JEPA-guided diffusion run.

The Kaggle notebook ``trnquanghuyn/jepa-guided-diffusion`` trains an adapter
that injects denoising I-JEPA latents into a frozen ADM 256x256 model, then
denoises the same noisy photos with ADM alone, the adapter given null latents
and the adapter given JEPA latents (same start, same noise draws). This script
reads that run's output directory and writes:

* ``paired_tests.csv``: each method contrast per noise condition: mean and
  median difference, paired bootstrap interval, images on which the first
  method is better, exact sign-flip p-value, Holm correction per contrast;
* ``per_image.csv``: per-image differences, JEPA reverse minus null reverse;
* ``diagnostics.json``: validation history, diffusion-schedule facts and
  adapter weight / latent-sensitivity diagnostics;
* ``jepa_minus_null.png``: the per-image differences, one point per image;
* with ``--comparison-output`` (the ``outputs`` directory of the
  ``jepa-denoise-comparison`` notebook, whose inputs are verified identical to
  this run's): ``standard_denoisers_tests.csv``, the same paired tests for each
  standard denoiser against ADM Tweedie and ADM reverse, and
  ``retained_noise.csv``, the share of the input noise left in every saved
  output and the part of its squared error uncorrelated with that noise;
* with ``--improvement-output`` (the ``outputs`` directory of the
  ``adm-noise-level-and-averaging`` notebook): ``improvement_tests.csv``, the
  same paired tests for the inference-only changes (start step from the
  estimated noise level, mean of 8 reverse samples) against v1's rule and the
  standard denoisers;
* with ``--edits-output`` (the ``outputs`` directory of the
  ``jepa-adapter-weight-edits`` notebook): ``weight_edit_tests.csv``, the same
  paired tests for the hand-edited adapters E2 and E3 against the null latents
  and the original JEPA latents.

    kaggle kernels output trnquanghuyn/jepa-guided-diffusion -p kaggle_output/jepa-guided-diffusion/output
    kaggle kernels output trnquanghuyn/jepa-denoise-comparison -p kaggle_output/denoise-comparison/output
    kaggle kernels output trnquanghuyn/adm-noise-level-and-averaging -p kaggle_output/adm-noise-level/output
    kaggle kernels output trnquanghuyn/jepa-adapter-weight-edits -p kaggle_output/wsp-edits/output
    python analysis/jepa_guided_diffusion.py \
        --kaggle-output kaggle_output/jepa-guided-diffusion/output/outputs \
        --comparison-output kaggle_output/denoise-comparison/output/outputs \
        --improvement-output kaggle_output/adm-noise-level/output/outputs \
        --edits-output kaggle_output/wsp-edits/output/outputs \
        --output-directory results/jepa_guided_diffusion
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from laboratory.evaluation.statistics import holm_adjust, paired_bootstrap_difference
from jepa_adapter_weights import adapter_diagnostics

METRICS = ("psnr", "ssim", "lpips")
LOWER_IS_BETTER = {"lpips"}
CONTRASTS = {
    "JEPA reverse - null reverse": ("adapter JEPA: reverse", "adapter null: reverse"),
    "null reverse - ADM reverse": ("adapter null: reverse", "ADM reverse"),
    "JEPA reverse - ADM reverse": ("adapter JEPA: reverse", "ADM reverse"),
    "JEPA Tweedie - ADM Tweedie": ("adapter JEPA: Tweedie", "ADM Tweedie"),
    "ADM Tweedie - ADM reverse": ("ADM Tweedie", "ADM reverse"),
}
POOLED = "gaussian, per-image mean"
IMPROVEMENT_CONTRASTS = {
    "Tweedie: σ̂ rule - v1 rule": ("ADM Tweedie (σ̂)", "ADM Tweedie"),
    "Tweedie: residual σ - v1 rule": ("ADM Tweedie (residual σ, oracle)", "ADM Tweedie"),
    "mean of 8 (σ̂) - Tweedie (σ̂)": ("ADM mean of 8 (σ̂)", "ADM Tweedie (σ̂)"),
    "1 sample: σ̂ rule - v1 rule": ("ADM reverse, 1 sample (σ̂)", "ADM reverse"),
    "Tweedie (σ̂) - Restormer": ("ADM Tweedie (σ̂)", "Restormer"),
    "Tweedie (residual σ) - Restormer": ("ADM Tweedie (residual σ, oracle)", "Restormer"),
    "Tweedie (σ̂) - SCUNet": ("ADM Tweedie (σ̂)", "SCUNet"),
    "Tweedie (σ̂) - DRUNet (σ̂)": ("ADM Tweedie (σ̂)", "DRUNet (σ̂)"),
    "DRUNet: σ̂ - nominal σ": ("DRUNet (σ̂)", "DRUNet"),
    "BM3D: σ̂ - nominal σ": ("BM3D (σ̂)", "BM3D"),
}
EDIT_CONTRASTS = {
    "JEPA - null (reverse)": ("JEPA: reverse", "null: reverse"),
    "E2 - null (reverse)": ("JEPA E2: reverse", "null: reverse"),
    "E3 - null (reverse)": ("JEPA E3: reverse", "null: reverse"),
    "E2 - JEPA (reverse)": ("JEPA E2: reverse", "JEPA: reverse"),
    "E3 - JEPA (reverse)": ("JEPA E3: reverse", "JEPA: reverse"),
    "E3 - E2 (reverse)": ("JEPA E3: reverse", "JEPA E2: reverse"),
}
VALIDATION_TIMESTEPS = (30, 60, 120, 200, 300, 500, 700, 900)  # VAL_T in the notebook
ALPHA_BAR = np.cumprod(1.0 - np.linspace(1e-4, 0.02, 1000, dtype=np.float64))  # ADM's linear schedule


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kaggle-output", type=Path, required=True,
                        help="directory with scores.csv, results.json and jepa_adapter.pt")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--comparison-output", type=Path,
                        help="outputs directory of the jepa-denoise-comparison notebook")
    parser.add_argument("--improvement-output", type=Path,
                        help="outputs directory of the adm-noise-level-and-averaging notebook")
    parser.add_argument("--edits-output", type=Path,
                        help="outputs directory of the jepa-adapter-weight-edits notebook")
    return parser.parse_args()


def load_scores(path: Path) -> tuple[dict[tuple[str, str, str], dict[str, float]], list[str]]:
    """``{(condition, method, metric): {image: value}}`` and the conditions in file order."""
    scores: dict[tuple[str, str, str], dict[str, float]] = {}
    conditions: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["noise"] not in conditions:
                conditions.append(row["noise"])
            for metric in METRICS:
                scores.setdefault((row["noise"], row["method"], metric), {})[row["image"]] = float(row[metric])
    return scores, conditions


def paired_values(scores, conditions, condition, first, second, metric) -> tuple[np.ndarray, np.ndarray]:
    """Per-image values aligned by image; the pooled condition averages each image over the Gaussian levels."""
    levels = [c for c in conditions if c.startswith("gaussian")] if condition == POOLED else [condition]
    images = sorted(scores[(levels[0], first, metric)])
    a = np.array([np.mean([scores[(level, first, metric)][image] for level in levels]) for image in images])
    b = np.array([np.mean([scores[(level, second, metric)][image] for level in levels]) for image in images])
    return a, b


def sign_flip_p_value(differences: np.ndarray) -> float:
    """Exact two-sided permutation p-value for ``mean(differences) = 0`` over all 2^n sign patterns."""
    n = differences.size
    if n > 20:
        raise ValueError(f"the exact test enumerates 2^n sign patterns; n={n} is too many")
    signs = 1.0 - 2.0 * ((np.arange(2**n)[:, None] >> np.arange(n)) & 1)
    null = np.abs(signs @ differences) / n
    return float(np.mean(null >= abs(differences.mean()) - 1e-12))


def paired_tests(scores, conditions, contrasts: dict[str, tuple[str, str]]) -> list[dict]:
    rows = []
    for name, (first, second) in contrasts.items():
        family = []
        for condition in [*conditions, POOLED]:
            for metric in METRICS:
                a, b = paired_values(scores, conditions, condition, first, second, metric)
                differences = a - b
                low, high = paired_bootstrap_difference(a, b)
                better = differences < 0 if metric in LOWER_IS_BETTER else differences > 0
                family.append(dict(
                    contrast=name, condition=condition, metric=metric, n=differences.size,
                    mean_difference=float(differences.mean()), median_difference=float(np.median(differences)),
                    ci_low=low, ci_high=high, first_better=int(better.sum()),
                    p_sign_flip=sign_flip_p_value(differences),
                ))
        for row, adjusted in zip(family, holm_adjust([row["p_sign_flip"] for row in family])):
            row["p_holm"] = adjusted
        rows += family
    return rows


def per_image(scores, conditions) -> list[dict]:
    first, second = CONTRASTS["JEPA reverse - null reverse"]
    rows = []
    for condition in conditions:
        for image in sorted(scores[(condition, first, "psnr")]):
            row = dict(condition=condition, image=image)
            for metric in METRICS:
                row[f"{metric}_difference"] = scores[(condition, first, metric)][image] - scores[(condition, second, metric)][image]
            row["null_reverse_psnr"] = scores[(condition, second, "psnr")][image]
            rows.append(row)
    return rows


def signal_to_noise(step: int) -> float:
    return float(ALPHA_BAR[step] / (1.0 - ALPHA_BAR[step]))


def start_step(noise_level: float) -> int:
    """The notebook's ``t_for_sigma``: pixel noise on [0, 1] doubles on ADM's [-1, 1] scale."""
    return int(np.abs(ALPHA_BAR - 1.0 / (1.0 + (2.0 * noise_level) ** 2)).argmin())


def schedule_facts(scores, conditions) -> dict:
    starts = {}
    for condition in conditions:
        if condition.startswith("gaussian"):
            levels = {condition: float(condition.split()[-1])}
        else:  # real noise: RMSE of the noisy input, recovered from its PSNR
            levels = {f"{condition}: {image}": 10 ** (-psnr / 20)
                      for image, psnr in scores[(condition, "noisy input", "psnr")].items()}
        for label, level in levels.items():
            step = start_step(level)
            starts[label] = dict(noise_level=level, start_step=step, signal_to_noise=signal_to_noise(step))
    # An exact posterior sample has twice the MSE of the exact posterior mean (3.01 dB).
    ratios = {}
    for condition in conditions:
        tweedie, reverse = scores[(condition, "ADM Tweedie", "psnr")], scores[(condition, "ADM reverse", "psnr")]
        ratio = np.array([10 ** ((tweedie[image] - reverse[image]) / 10) for image in tweedie])
        ratios[condition] = dict(geometric_mean=float(np.exp(np.log(ratio).mean())),
                                 minimum=float(ratio.min()), maximum=float(ratio.max()))
    return dict(start=starts, validation_signal_to_noise={str(t): signal_to_noise(t) for t in VALIDATION_TIMESTEPS},
                reverse_over_tweedie_mse=ratios)


def retained_noise(images: Path) -> list[dict]:
    """Per saved case and method: the share of the input noise left in the output (least-squares coefficient of
    output - clean on noisy - clean) and the part of the squared error uncorrelated with that noise.

    Each case directory holds ``00-clean-reference.png``, ``01-noisy-input.png`` and one ``NN-<method>.png`` per method.
    """
    from PIL import Image

    def load(path: Path) -> np.ndarray:
        return np.asarray(Image.open(path), dtype=np.float64) / 255.0

    rows = []
    for folder in sorted(path for path in images.iterdir() if path.is_dir()):
        clean_path, noisy_path, *method_paths = sorted(folder.glob("*.png"))
        clean = load(clean_path)
        noise = load(noisy_path) - clean
        for path in method_paths:
            residual = load(path) - clean
            kept = float((residual * noise).sum() / (noise * noise).sum())
            rows.append(dict(case=folder.name, method=path.stem[3:], kept_fraction=kept, mse=float((residual**2).mean()),
                             uncorrelated_mse=float(((residual - kept * noise) ** 2).mean())))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{value:.6g}" if isinstance(value, float) else value for key, value in row.items()})


def plot_per_image(rows: list[dict], conditions: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    worst = min(rows, key=lambda row: row["psnr_difference"])["image"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, metric, label in zip(axes, ("psnr", "lpips"),
                                   ("Δ PSNR, dB (above 0: JEPA better)", "Δ LPIPS (below 0: JEPA better)")):
        for position, condition in enumerate(conditions):
            values = [row for row in rows if row["condition"] == condition]
            for offset, row in zip(np.linspace(-0.25, 0.25, len(values)), values):
                outlier = row["image"] == worst
                axis.scatter(position + offset, row[f"{metric}_difference"], s=24, zorder=3,
                             color="tab:red" if outlier else "tab:blue", label=worst if outlier else None)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(range(len(conditions)))
        axis.set_xticklabels(conditions)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[:1], labels[:1], frameon=False)
    figure.suptitle("JEPA latents minus null latents, reverse process, one point per image")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def extra_tests(scores_path: Path, conditions: list[str], contrasts: dict[str, tuple[str, str]], path: Path) -> None:
    """Paired tests for another run's scores on the same cases; prints the PSNR and LPIPS rows."""
    everything, _ = load_scores(scores_path)
    tests = paired_tests(everything, conditions, contrasts)
    write_csv(path, tests)
    for row in tests:
        if row["metric"] in ("psnr", "lpips"):
            print(f"{row['contrast']:34s} {row['condition']:25s} {row['metric']:5s} {row['mean_difference']:+.4f} "
                  f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]  better {row['first_better']}/{row['n']}  "
                  f"Holm p={row['p_holm']:.3g}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    args = parse_arguments()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    results = json.loads((args.kaggle_output / "results.json").read_text(encoding="utf-8"))
    config = results["config"]
    scores, conditions = load_scores(args.kaggle_output / "scores.csv")

    tests = paired_tests(scores, conditions, CONTRASTS)
    write_csv(args.output_directory / "paired_tests.csv", tests)
    images = per_image(scores, conditions)
    write_csv(args.output_directory / "per_image.csv", images)
    diagnostics = dict(
        training=dict(steps_planned=config["STEPS"], steps_done=config["steps_done"], validation=[
            dict(step=row["step"], adm=row["adm"], null=row["null"], jepa=row["jepa"], jepa_minus_null=row["jepa"] - row["null"])
            for row in results["validation"]]),
        schedule=schedule_facts(scores, conditions),
        adapter=adapter_diagnostics(args.kaggle_output / "jepa_adapter.pt", config["inject"], config["steps_done"], config["LR"]),
    )
    (args.output_directory / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    plot_per_image(images, conditions, args.output_directory / "jepa_minus_null.png")

    for row in tests:
        if row["contrast"] == "JEPA reverse - null reverse":
            print(f"JEPA - null  {row['condition']:25s} {row['metric']:5s} {row['mean_difference']:+.4f} "
                  f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]  JEPA better {row['first_better']}/{row['n']}  "
                  f"p={row['p_sign_flip']:.3g}  Holm p={row['p_holm']:.3g}")

    if args.comparison_output:
        standard, _ = load_scores(args.comparison_output / "standard_denoisers_scores.csv")
        methods = list(dict.fromkeys(method for _, method, _ in standard))
        contrasts = {f"{method} - {reference}": (method, reference)
                     for method in methods for reference in ("ADM Tweedie", "ADM reverse")}
        standard_tests = paired_tests({**scores, **standard}, conditions, contrasts)
        write_csv(args.output_directory / "standard_denoisers_tests.csv", standard_tests)
        for row in standard_tests:
            if row["contrast"].endswith("ADM Tweedie") and row["metric"] in ("psnr", "lpips"):
                print(f"{row['contrast']:24s} {row['condition']:25s} {row['metric']:5s} {row['mean_difference']:+.4f} "
                      f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]  better {row['first_better']}/{row['n']}  "
                      f"Holm p={row['p_holm']:.3g}")
        write_csv(args.output_directory / "retained_noise.csv", retained_noise(args.comparison_output / "images"))

    if args.improvement_output:
        extra_tests(args.improvement_output / "all_scores.csv", conditions, IMPROVEMENT_CONTRASTS,
                    args.output_directory / "improvement_tests.csv")
    if args.edits_output:
        extra_tests(args.edits_output / "scores.csv", conditions, EDIT_CONTRASTS, args.output_directory / "weight_edit_tests.csv")
    print(f"wrote {args.output_directory}")


if __name__ == "__main__":
    main()
