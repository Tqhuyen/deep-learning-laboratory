"""Weight-space reading of the JEPA adapter behind the hand edits of ``jepa-adapter-weight-edits``.

Reads the trained adapter (``jepa_adapter.pt`` from ``jepa-guided-diffusion``) and the forward-pass probes of
``jepa-adapter-weight-probe`` (Tweedie scores of each adapter path on 40 held-out images, the JEPA latents of
those images, ADM's time embedding for every step) and writes ``weight_reading.json``:

* the spectra of the four output layers (effective rank, energy in the top directions) and how much of the
  null-latent offset lies on their top directions;
* the latent-induced change Δg of the ``glob`` path: the share common to all images, the share along the
  time-embedding tangent (a plain timestep shift), and how well its image-specific part predicts the extra
  noise kept (principal components, and a ridge direction checked leave-one-image-out);
* the edits E2 (``glob`` ignores the latents) and E3 (rank-1 removal of the common offset from ``glob.2``):
  the change of the null-latent output, and for E3 the common offset left on images it was not fitted on.

No forward pass of ADM or of the encoder is run here; the probes ran on Kaggle.

    kaggle kernels output trnquanghuyn/jepa-adapter-weight-probe -p kaggle_output/wsp-probe/output
    python analysis/jepa_adapter_weights.py \
        --kaggle-output kaggle_output/jepa-guided-diffusion/output/outputs \
        --probe-output kaggle_output/wsp-probe/output/outputs \
        --output-directory results/jepa_guided_diffusion
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

INJECT = {9: (32, 512), 12: (16, 512), 15: (8, 1024)}
T_STAR = {0.1: 57, 0.2: 117}   # t_for_sigma of the probe noise levels
E3_RIDGE = 0.01


def _zero(module: nn.Module) -> nn.Module:
    nn.init.zeros_(module.weight)
    nn.init.zeros_(module.bias)
    return module


class JEPAAdapter(nn.Module):
    """The notebook's adapter (cell 10); attribute names match the saved state dict."""

    def __init__(self, latent_dimension: int, points: dict[int, tuple[int, int]], embedding_dimension: int):
        super().__init__()
        self.resolutions = {str(block): resolution for block, (resolution, _) in points.items()}
        self.branches = nn.ModuleDict({str(block): nn.Sequential(
            nn.Conv2d(latent_dimension, channels, 1), nn.GroupNorm(32, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(), _zero(nn.Conv2d(channels, channels, 1)),
        ) for block, (_, channels) in points.items()})
        self.glob = nn.Sequential(nn.Linear(latent_dimension, embedding_dimension), nn.SiLU(),
                                  _zero(nn.Linear(embedding_dimension, embedding_dimension)))

    def forward(self, latents: torch.Tensor) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        features = {}
        for block, branch in self.branches.items():
            size = self.resolutions[block]
            resized = latents if size == latents.shape[-1] else F.interpolate(
                latents, size=(size, size), mode="bilinear", align_corners=False)
            features[int(block)] = branch(resized)
        return features, self.glob(latents.mean((2, 3)))


def _rms(tensor: torch.Tensor) -> float:
    return float(tensor.pow(2).mean().sqrt())


def _layer_normed(latents: torch.Tensor) -> torch.Tensor:
    """The notebook LayerNorms every token (no affine) before the adapter sees it."""
    return F.layer_norm(latents.permute(0, 2, 3, 1), (latents.shape[1],)).permute(0, 3, 1, 2)


def adapter_diagnostics(path: Path, inject: dict[str, list[int]], steps_done: int, learning_rate: float) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    points = {int(block): (resolution, channels) for block, (resolution, channels) in inject.items()}
    latent_dimension = state[f"branches.{next(iter(inject))}.0.weight"].shape[1]
    embedding_dimension = state["glob.2.weight"].shape[0]
    adapter = JEPAAdapter(latent_dimension, points, embedding_dimension).eval()
    adapter.load_state_dict(state)

    layers = {f"block {block}": dict(
        output_weight_rms=_rms(state[f"branches.{block}.5.weight"]),
        output_bias_rms=_rms(state[f"branches.{block}.5.bias"]),
        group_norm_scale_change_rms=_rms(state[f"branches.{block}.1.weight"] - 1.0),
        group_norm_shift_rms=_rms(state[f"branches.{block}.1.bias"]),
        default_initialisation_weight_rms=1.0 / math.sqrt(3.0 * channels),
    ) for block, (_, channels) in points.items()}
    layers["time embedding"] = dict(
        output_weight_rms=_rms(state["glob.2.weight"]), output_bias_rms=_rms(state["glob.2.bias"]),
        default_initialisation_weight_rms=1.0 / math.sqrt(3.0 * embedding_dimension))

    generator = torch.Generator().manual_seed(0)
    shape = (8, latent_dimension)
    probes = {  # random LayerNormed latents: a scale for how strongly the output depends on the input
        "independent tokens": _layer_normed(torch.randn(*shape, 16, 16, generator=generator)),
        "smooth field": _layer_normed(F.interpolate(torch.randn(*shape, 4, 4, generator=generator),
                                                    size=(16, 16), mode="bicubic", align_corners=False)),
        "one token everywhere": _layer_normed(torch.randn(*shape, 1, 1, generator=generator).expand(-1, -1, 16, 16)),
    }
    with torch.no_grad():
        null_features, null_embedding = adapter(torch.zeros(1, latent_dimension, 16, 16))
        null = {f"block {block}": dict(rms=_rms(feature),
                                       interior_spatial_std=float(feature[0, :, 1:-1, 1:-1].std(dim=(1, 2)).max()))
                for block, feature in null_features.items()}
        null["time embedding"] = dict(rms=_rms(null_embedding))
        sensitivity = {}
        for name, latents in probes.items():
            features, embedding = adapter(latents)
            entry = {f"block {block}": dict(
                change_rms=_rms(features[block] - null_features[block]),
                relative_to_null=_rms(features[block] - null_features[block]) / _rms(null_features[block]),
                between_latents_rms=_rms(features[block][:4] - features[block][4:]),
            ) for block in points}
            entry["time embedding"] = dict(change_rms=_rms(embedding - null_embedding),
                                           relative_to_null=_rms(embedding - null_embedding) / _rms(null_embedding))
            sensitivity[name] = entry
    return dict(parameters=sum(value.numel() for value in state.values()),
                adam_displacement_bound=learning_rate * steps_done,
                layers=layers, null_latents=null, latent_sensitivity=sensitivity)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kaggle-output", type=Path, required=True, help="jepa-guided-diffusion outputs (jepa_adapter.pt)")
    parser.add_argument("--probe-output", type=Path, required=True, help="jepa-adapter-weight-probe outputs")
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def spectra(state: dict, null_offsets: dict[str, torch.Tensor]) -> dict:
    """Effective rank and top-direction energy of each output layer, and the null offset's share on its top directions."""
    layers = {f"block {b}": state[f"branches.{b}.5.weight"][:, :, 0, 0] for b in INJECT} | {"glob": state["glob.2.weight"]}
    out = {}
    for name, weight in layers.items():
        u, s, _ = torch.linalg.svd(weight.double(), full_matrices=False)
        energy = s**2 / (s**2).sum()
        null = null_offsets[name].double() / null_offsets[name].double().norm()
        on_top = u[:, :5].T @ null
        out[name] = dict(effective_rank=float(torch.exp(-(energy * torch.log(energy + 1e-30)).sum())),
                         energy_top1=float(energy[0]), energy_top5=float(energy[:5].sum()), energy_top20=float(energy[:20].sum()),
                         null_offset_on_top5=float(on_top.pow(2).sum()), null_offset_cosine_top1=float(on_top[0].abs()))
    return out


def load_probe(probe_dir: Path):
    """Probe latents keyed by (image, sigma) and the Tweedie kept-noise change of the glob-only variant against null."""
    latents = {(int(k.split("|")[0]), float(k.split("|")[1])): v.float()
               for k, v in torch.load(probe_dir / "latents.pt", map_location="cpu", weights_only=True).items()}
    kept: dict[tuple[int, float], dict[str, float]] = {}
    with (probe_dir / "probe_tweedie.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            kept.setdefault((int(row["image"]), float(row["sigma"])), {})[row["variant"]] = float(row["kept"])
    d_kept = {key: kept[key]["glob only"] - kept[key]["null"] for key in latents}
    return latents, d_kept


def ridge_direction(x: torch.Tensor, y: torch.Tensor, lam: float = 1e-4) -> torch.Tensor:
    xc = x - x.mean(0)
    return torch.linalg.solve(xc.T @ xc + lam * torch.eye(x.shape[1], dtype=x.dtype), xc.T @ (y - y.mean()))


def r_squared(features: torch.Tensor, y: torch.Tensor) -> float:
    design = torch.cat([torch.ones(len(y), 1, dtype=y.dtype), features], 1)
    beta = torch.linalg.lstsq(design, y[:, None]).solution
    residual = y[:, None] - design @ beta
    return float(1 - residual.pow(2).sum() / (y - y.mean()).pow(2).sum())


def e3_update(w2: torch.Tensor, hidden: torch.Tensor, h0: torch.Tensor, ridge: float) -> tuple[torch.Tensor, torch.Tensor]:
    """m = mean latent-induced output offset; v with v·h0 = 0 and v·h ≈ 1 for latent-conditioned hidden states."""
    m = w2 @ (hidden.mean(0) - h0)
    p_perp = torch.eye(len(h0), dtype=h0.dtype) - torch.outer(h0, h0) / (h0 @ h0)
    a = hidden @ p_perp
    v = p_perp @ torch.linalg.solve(a.T @ a + ridge * torch.eye(len(h0), dtype=h0.dtype), a.T @ torch.ones(len(hidden), dtype=h0.dtype))
    return m, v


def glob_reading(state: dict, latents: dict, d_kept: dict, embedding: torch.Tensor) -> dict:
    w1, b1, w2, b2 = (state[k].double() for k in ("glob.0.weight", "glob.0.bias", "glob.2.weight", "glob.2.bias"))
    keys = sorted(latents)
    hidden = F.silu(torch.stack([latents[k].mean((2, 3))[0].double() for k in keys]) @ w1.T + b1)
    h0 = F.silu(b1)
    g0 = w2 @ h0 + b2
    delta = hidden @ w2.T + b2 - g0
    y = torch.tensor([d_kept[k] for k in keys], dtype=torch.float64)
    sigma = torch.tensor([s for _, s in keys], dtype=torch.float64)[:, None]

    reading = {"common_share": float(delta.mean(0).norm() / delta.norm(dim=1).pow(2).mean().sqrt())}
    for s, t in T_STAR.items():
        tangent = (embedding[t + 1] - embedding[t - 1]).double() / 2
        rows = [i for i, k in enumerate(keys) if k[1] == s]
        shift = delta[rows] @ tangent / (tangent @ tangent)
        reading[f"sigma {s}"] = dict(
            null_offset_timestep_shift=float(g0 @ tangent / (tangent @ tangent)),
            timestep_shift_mean=float(shift.mean()), timestep_shift_min=float(shift.min()), timestep_shift_max=float(shift.max()),
            share_on_tangent=float(((delta[rows] @ tangent) ** 2 / (tangent @ tangent) / delta[rows].pow(2).sum(1)).mean()),
            corr_shift_kept=float(np.corrcoef(shift, y[rows])[0, 1]))
    _, _, vt = torch.linalg.svd(delta - delta.mean(0), full_matrices=False)
    pcs = (delta - delta.mean(0)) @ vt[:6].T
    reading["r2_sigma_only"] = r_squared(sigma, y)
    reading["r2_sigma_and_pcs"] = {m: r_squared(torch.cat([sigma, pcs[:, :m]], 1), y) for m in (1, 3, 6)}
    predicted = torch.zeros_like(y)
    for image in sorted({i for i, _ in keys}):
        train = torch.tensor([i != image for i, _ in keys])
        w = ridge_direction(delta[train], y[train])
        predicted[~train] = (delta[~train] - delta[train].mean(0)) @ w + y[train].mean()
    reading["ridge_leave_one_image_out_corr"] = float(np.corrcoef(predicted, y)[0, 1])

    m, v = e3_update(w2, hidden, h0, E3_RIDGE)
    w2_e3 = w2 - torch.outer(m, v)
    images = sorted({i for i, _ in keys})
    fit = torch.tensor([i in images[::2] for i, _ in keys])
    m_half, v_half = e3_update(w2, hidden[fit], h0, E3_RIDGE)
    held = hidden[~fit] @ (w2 - torch.outer(m_half, v_half)).T + b2 - g0
    reading["edits"] = dict(
        e2_null_change=0.0,   # glob.0.weight = 0 leaves h(0) = SiLU(b1) and therefore g(0) untouched
        e3_null_change=float((w2_e3 @ h0 - w2 @ h0).abs().max()),
        e3_relative_update=float(torch.outer(m, v).norm() / w2.norm()),
        e3_v_dot_h_mean=float((hidden @ v).mean()), e3_v_dot_h_std=float((hidden @ v).std()),
        common_offset_rms_before=float(delta.mean(0).pow(2).mean().sqrt()),
        common_offset_rms_after_in_sample=float((hidden @ w2_e3.T + b2 - g0).mean(0).pow(2).mean().sqrt()),
        common_offset_rms_held_out_half_before=float(delta[~fit].mean(0).pow(2).mean().sqrt()),
        common_offset_rms_held_out_half_after=float(held.mean(0).pow(2).mean().sqrt()))
    return reading


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    args = parse_arguments()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.kaggle_output / "jepa_adapter.pt", map_location="cpu", weights_only=True)
    adapter = JEPAAdapter(1280, INJECT, state["glob.2.weight"].shape[0]).eval()
    adapter.load_state_dict(state)
    with torch.no_grad():
        features, offset = adapter(torch.zeros(1, 1280, 16, 16))
    null_offsets = {f"block {b}": features[b][0, :, features[b].shape[-1] // 2, features[b].shape[-1] // 2] for b in INJECT}
    null_offsets["glob"] = offset[0]

    latents, d_kept = load_probe(args.probe_output)
    embedding = torch.load(args.probe_output / "time_embedding.pt", map_location="cpu", weights_only=True)["emb"]
    reading = dict(spectra=spectra(state, null_offsets), glob=glob_reading(state, latents, d_kept, embedding))
    (args.output_directory / "weight_reading.json").write_text(json.dumps(reading, indent=2), encoding="utf-8")
    print(json.dumps(reading, indent=1))
    print(f"wrote {args.output_directory / 'weight_reading.json'}")


if __name__ == "__main__":
    main()
