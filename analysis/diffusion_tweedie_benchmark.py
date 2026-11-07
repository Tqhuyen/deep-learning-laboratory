"""Paired statistics for the one-step diffusion posterior-mean benchmark.

The Kaggle notebook ``trnquanghuyn/diffusion-tweedie-benchmark`` (``notebooks/diffusion_tweedie_benchmark.ipynb``)
denoises CBSD68, Kodak24 and McMaster at σ ∈ {15, 25, 50} under DMID's protocol with frozen ADM 256×256 variants
and supervised baselines, and scores DMID's released outputs with the same code. This script reads its ``outputs``
directory and writes:

* ``means.csv``: mean PSNR/SSIM/LPIPS per dataset, σ and method, with the published PSNR of DMID-d, Restormer,
  SwinIR, DRUNet and MambaIR;
* ``paired_tests.csv``: each contrast per dataset and σ and pooled over the datasets per σ: mean and median
  difference, paired bootstrap interval, images on which the first method is better, sign-flip p-value (exact for
  n ≤ 20, otherwise 200,000 random sign patterns), Holm correction within each contrast;
* ``calibration.csv``: held-out PSNR per start-step factor f and σ, the per-image oracle gain over the chosen f;
* ``fusion.csv``: blend PSNR per weight, the best global weight per set and the per-image oracle;
* ``ensemble.png``, ``calibration.png``: the ensemble-size and start-step curves.

    kaggle kernels output trnquanghuyn/diffusion-tweedie-benchmark -p kaggle_output/sota/runA
    python analysis/diffusion_tweedie_benchmark.py --run-output kaggle_output/sota/runA/outputs \\
        --output-directory results/diffusion_benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from laboratory.evaluation.statistics import holm_adjust, paired_bootstrap_difference

METRICS = ("psnr", "ssim", "lpips")
LOWER_IS_BETTER = {"lpips"}
SIGMAS = (15, 25, 50)
DMID = "DMID-d (released outputs)"
CONTRASTS = {
    "one step, DMID t − DMID-d (1500 evaluations)": ("ADM one step, DMID t", DMID),
    "continuous t(σ) − DMID t": ("ADM one step, t(σ)", "ADM one step, DMID t"),
    "calibrated f − t(σ)": ("ADM-cal ×1", "ADM one step, t(σ)"),
    "×2 − ×1": ("ADM-cal ×2", "ADM-cal ×1"),
    "×4 − ×2": ("ADM-cal ×4", "ADM-cal ×2"),
    "×8 − ×4": ("ADM-cal ×8", "ADM-cal ×4"),
    "two steps (η=0) − ×1": ("ADM-cal two steps (η=0)", "ADM-cal ×1"),
    "256-px tiles − whole image": ("ADM-cal ×1, 256-px tiles", "ADM-cal ×1"),
    "ADM-cal ×8 − DMID-d": ("ADM-cal ×8", DMID),
    "ADM-cal ×8 − Restormer": ("ADM-cal ×8", "Restormer"),
    "ADM-cal ×8 − SwinIR": ("ADM-cal ×8", "SwinIR"),
    "ADM-cal ×8 − SCUNet": ("ADM-cal ×8", "SCUNet"),
    "ADM-cal ×8 − DRUNet": ("ADM-cal ×8", "DRUNet"),
    "ADM blind ×1 − Restormer blind": ("ADM blind ×1", "Restormer blind"),
    "ADM blind ×1 − ADM-cal ×1 (cost of not knowing σ)": ("ADM blind ×1", "ADM-cal ×1"),
}
PUBLISHED = {  # PSNR at σ = 15, 25, 50 (DMID Table 1; MambaIR ECCV 2024, σ = 25 on CBSD68 unreadable in the PDF)
    "DMID-d (published)": {"CBSD68": (34.45, 31.86, 28.72), "Kodak24": (35.51, 33.12, 30.14), "McMaster": (35.72, 33.49, 30.50)},
    "Restormer (published)": {"CBSD68": (34.40, 31.79, 28.60), "Kodak24": (35.47, 33.04, 30.01), "McMaster": (35.61, 33.34, 30.30)},
    "SwinIR (published)": {"CBSD68": (34.42, 31.78, 28.56), "Kodak24": (35.34, 32.89, 29.79), "McMaster": (35.61, 33.20, 30.22)},
    "DRUNet (published)": {"CBSD68": (34.30, 31.69, 28.51), "Kodak24": (35.31, 32.89, 29.86), "McMaster": (35.40, 33.14, 30.08)},
    "MambaIR (published)": {"CBSD68": (34.48, None, 28.66), "Kodak24": (35.42, 32.99, 29.92), "McMaster": (35.70, 33.43, 30.35)},
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-output", type=Path, required=True, help="outputs directory of diffusion-tweedie-benchmark")
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def sign_flip_p_value(differences: np.ndarray, samples: int = 200_000, seed: int = 0) -> float:
    """Two-sided sign-flip p-value for ``mean(differences) = 0``: exact for n ≤ 20, Monte Carlo otherwise."""
    n = differences.size
    if n <= 20:
        signs = 1.0 - 2.0 * ((np.arange(2**n)[:, None] >> np.arange(n)) & 1)
    else:
        signs = np.random.default_rng(seed).choice((-1.0, 1.0), size=(samples, n))
    null = np.abs(signs @ differences) / n
    return float((np.sum(null >= abs(differences.mean()) - 1e-12) + (n > 20)) / (len(null) + (n > 20)))


def load(run: Path) -> pd.DataFrame:
    scores = pd.read_csv(run / "scores.csv")
    dmid = pd.read_csv(run / "dmid_released_scores.csv")
    return pd.concat([scores, dmid[dmid.dataset.isin(scores.dataset.unique())]], ignore_index=True)


def means(scores: pd.DataFrame) -> pd.DataFrame:
    table = scores.groupby(["dataset", "sigma", "method"])[list(METRICS)].mean().reset_index()
    published = [dict(dataset=ds, sigma=s, method=name, psnr=v)
                 for name, sets in PUBLISHED.items() for ds, values in sets.items() if ds in set(table.dataset)
                 for s, v in zip(SIGMAS, values) if v is not None]
    return pd.concat([table, pd.DataFrame(published)], ignore_index=True)


def paired_tests(scores: pd.DataFrame, contrasts: dict[str, tuple[str, str]] = CONTRASTS) -> pd.DataFrame:
    wide = scores.pivot_table(index=["dataset", "sigma", "image"], columns="method", values=list(METRICS))
    datasets = sorted(scores.dataset.unique())
    rows = []
    for name, (first, second) in contrasts.items():
        if first not in wide["psnr"] or second not in wide["psnr"]:
            continue
        family = []
        for ds in [*datasets, "all"]:
            for s in SIGMAS:
                for metric in METRICS:
                    part = wide[metric] if ds == "all" else wide[metric].xs(ds, level="dataset", drop_level=False)
                    part = part.xs(s, level="sigma")[[first, second]].dropna()
                    if part.empty:
                        continue
                    a, b = part[first].to_numpy(), part[second].to_numpy()
                    d = a - b
                    low, high = paired_bootstrap_difference(a, b)
                    better = d < 0 if metric in LOWER_IS_BETTER else d > 0
                    family.append(dict(contrast=name, dataset=ds, sigma=s, metric=metric, n=d.size, mean_difference=float(d.mean()),
                                       median_difference=float(np.median(d)), ci_low=low, ci_high=high,
                                       first_better=int(better.sum()), p_sign_flip=sign_flip_p_value(d)))
        for row, adjusted in zip(family, holm_adjust([r["p_sign_flip"] for r in family])):
            row["p_holm"] = adjusted
        rows += family
    return pd.DataFrame(rows)


def calibration(run: Path, summary: dict) -> pd.DataFrame:
    cal = pd.read_csv(run / "calibration.csv")
    f = cal[cal.kind == "f"]
    curve = f.groupby(["sigma", "value"]).psnr.mean().rename("mean_psnr").reset_index()
    best = f.groupby(["sigma", "image"]).psnr.max().groupby("sigma").mean()
    for s in SIGMAS:
        grid = curve[curve.sigma == s]
        chosen = summary["f_hat"][str(s)]
        nearest = grid.iloc[(grid.value - chosen).abs().argmin()]
        curve.loc[curve.sigma == s, "f_hat"] = chosen
        curve.loc[curve.sigma == s, "per_image_oracle_gain"] = best[s] - nearest.mean_psnr
    return curve


def fusion(run: Path) -> pd.DataFrame:
    fu = pd.read_csv(run / "fusion.csv")
    by_w = fu.groupby(["dataset", "sigma", "partner", "w"]).psnr.mean().rename("mean_psnr").reset_index()
    oracle = fu.groupby(["dataset", "sigma", "partner", "image"]).psnr.max().groupby(["dataset", "sigma", "partner"]).mean()
    by_w["per_image_oracle"] = [oracle[(d, s, p)] for d, s, p in zip(by_w.dataset, by_w.sigma, by_w.partner)]
    return by_w


def plot(scores: pd.DataFrame, curve: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = scores.groupby(["dataset", "sigma", "method"]).psnr.mean()
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for axis, s in zip(axes, SIGMAS):
        for ds in sorted(scores.dataset.unique()):
            ks = [k for k in (1, 2, 4, 8) if (ds, s, f"ADM-cal ×{k}") in m.index]
            line, = axis.plot(ks, [m[(ds, s, f"ADM-cal ×{k}")] - m[(ds, s, "ADM-cal ×1")] for k in ks], "o-", label=ds)
            if (ds, s, DMID) in m.index:
                axis.axhline(m[(ds, s, DMID)] - m[(ds, s, "ADM-cal ×1")], ls=":", lw=1, color=line.get_color())
        axis.set(xscale="log", title=f"σ = {s}", xlabel="symmetry transforms averaged", ylabel="Δ PSNR vs one pass, dB")
        axis.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
        axis.minorticks_off()
    axes[0].plot([], [], "k:", lw=1, label="DMID-d (1500 evaluations)")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "ensemble.png", dpi=120)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(5, 3.6))
    for s, g in curve.groupby("sigma"):
        axis.plot(g.value, g.mean_psnr - g.mean_psnr.max(), "o-", label=f"σ = {s}")
    axis.set(xlabel="start-step factor f (t matched to f·σ)", ylabel="held-out PSNR − best, dB", title="calibration (DIV2K)")
    axis.legend()
    plt.tight_layout()
    plt.savefig(out / "calibration.png", dpi=120)
    plt.close(figure)


PANELS = {"clean": "clean", "noisy": "noisy", "Restormer": "restormer", "SwinIR": "swinir", "ADM one step, DMID t": "adm-one-step-dmid-t",
          "ADM-cal ×2": "adm-cal-2", "ADM-cal two steps (η=0)": "adm-cal-two-steps-0"}  # file names: NN-<slug>.png


def qualitative(run: Path, scores: pd.DataFrame, out: Path, crop: int = 192) -> None:
    """Centre crops of the first saved image of each dataset at σ = 50; titles carry the whole-image PSNR."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    folders = sorted(p for p in (run / "images").glob("*/000") if p.is_dir())
    if not folders:
        return
    psnr = scores[scores.sigma == 50].set_index(["dataset", "image", "method"]).psnr
    names = scores[scores.sigma == 50].groupby("dataset").image.first()
    figure, axes = plt.subplots(len(folders), len(PANELS), figsize=(2.2 * len(PANELS), 2.4 * len(folders)), squeeze=False)
    for row, folder in enumerate(folders):
        ds = folder.parent.name
        for col, (label, slug) in enumerate(PANELS.items()):
            img = np.asarray(Image.open(next(p for p in folder.glob("*.png") if p.stem[3:] == slug)))
            h, w = img.shape[:2]
            img = img[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]
            key = (ds, names[ds], "noisy input" if label == "noisy" else label)
            title = label if label == "clean" or key not in psnr.index else f"{label}\n{psnr[key]:.2f} dB"
            axes[row, col].imshow(img)
            axes[row, col].set_title(title, fontsize=7)
            axes[row, col].axis("off")
        axes[row, 0].set_title(f"{ds} {names[ds]}\nclean", fontsize=7)
    plt.tight_layout()
    plt.savefig(out / "qualitative_sigma50.png", dpi=110)
    plt.close(figure)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    args = parse_arguments()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.run_output / "summary.json").read_text(encoding="utf-8"))
    scores = load(args.run_output)
    means(scores).to_csv(args.output_directory / "means.csv", index=False, float_format="%.6g")
    tests = paired_tests(scores)
    tests.to_csv(args.output_directory / "paired_tests.csv", index=False, float_format="%.6g")
    curve = calibration(args.run_output, summary)
    curve.to_csv(args.output_directory / "calibration.csv", index=False, float_format="%.6g")
    fusion(args.run_output).to_csv(args.output_directory / "fusion.csv", index=False, float_format="%.6g")
    plot(scores, curve, args.output_directory)
    qualitative(args.run_output, scores, args.output_directory)
    psnr = tests[tests.metric == "psnr"]
    for name in CONTRASTS:
        for _, r in psnr[psnr.contrast == name].iterrows():
            print(f"{name:48s} {r.dataset:9s} σ{r.sigma:<3d} {r.mean_difference:+.3f} dB [{r.ci_low:+.3f}, {r.ci_high:+.3f}] "
                  f"better {r.first_better}/{r.n}  Holm p={r.p_holm:.3g}")
    print(f"wrote {args.output_directory}")


if __name__ == "__main__":
    main()
