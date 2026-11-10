"""Evaluate MP-HSIR checkpoints on identical noisy inputs.

Synthetic: every test crop x condition gets noise from a generator seeded by (set, image, condition), so all
checkpoints see the same noisy cube. PSNR and SSIM are per-band means with images clipped to [0, 1], matching
MP-HSIR ``utils/val_utils.compute_psnr_ssim`` (SSIM: 7x7 uniform window, sample covariance, K1=0.01, K2=0.03).
SAM is the mean spectral angle in degrees.

Real (HyRank Hyperion, no clean target): stripe index (detrended column-mean variation relative to band std),
robust noise sigma (MAD of along-track differences) and SAM between output and input.

    python hsi_mphsir/evaluate.py --data /workspace/hsi_data --checkpoint official --out runs/eval_official.csv
"""

import argparse
import csv
import json
import zlib
from pathlib import Path

import model as mphsir
import noise
import numpy as np
import torch
import torch.nn.functional as F

import data as hsi_data

PAPER = {
    ("WDC", "case1_noniid"): 35.21,
    ("WDC", "case2_stripe"): 34.99,
    ("WDC", "case3_deadline"): 34.72,
    ("WDC", "case4_mixture"): 31.36,
    ("Xiongan", "gauss30"): 40.55,
    ("Xiongan", "gauss50"): 38.70,
    ("Xiongan", "gauss70"): 37.17,
}


def psnr_per_band(restored, clean):
    mse = ((restored - clean) ** 2).flatten(2).mean(-1).clamp(min=1e-12)
    return (10 * torch.log10(1.0 / mse)).mean().item()


def ssim_per_band(restored, clean, window=7):
    c1, c2 = 0.01**2, 0.03**2
    cov_norm = window**2 / (window**2 - 1)
    pool = lambda t: F.avg_pool2d(t, window, stride=1)
    ux, uy = pool(restored), pool(clean)
    vx = cov_norm * (pool(restored * restored) - ux * ux)
    vy = cov_norm * (pool(clean * clean) - uy * uy)
    vxy = cov_norm * (pool(restored * clean) - ux * uy)
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux**2 + uy**2 + c1) * (vx + vy + c2))
    return s.flatten(2).mean(-1).mean().item()


def sam_degrees(a, b):
    dot = (a * b).sum(1)
    norm = a.norm(dim=1) * b.norm(dim=1)
    return (
        torch.rad2deg(torch.acos((dot / norm.clamp(min=1e-8)).clamp(-1, 1)))
        .mean()
        .item()
    )


def stripe_index(cube):
    """Mean over bands of std(column mean - its 9-column moving average) / std(band)."""
    column_mean = cube.mean(2)
    smooth = F.avg_pool1d(column_mean, 9, stride=1, padding=4, count_include_pad=False)
    return (
        ((column_mean - smooth).std(-1) / cube.flatten(2).std(-1).clamp(min=1e-8))
        .mean()
        .item()
    )


def noise_sigma(cube):
    diff = (cube[:, :, 1:, :] - cube[:, :, :-1, :]).flatten(2)
    mad = (diff - diff.median(-1, keepdim=True).values).abs().median(-1).values
    return (mad / 0.6745 / np.sqrt(2)).median().item()


@torch.no_grad()
def restore(net, noisy, prompt, tile=256, overlap=32):
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            return net(noisy, prompt).float()
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
    _, _, rows, columns = noisy.shape
    out = torch.zeros_like(noisy)
    weight = torch.zeros_like(noisy[:, :1])
    step = tile - overlap
    for r in range(0, max(rows - overlap, 1), step):
        for c in range(0, max(columns - overlap, 1), step):
            r0, c0 = min(r, rows - tile), min(c, columns - tile)
            with torch.autocast("cuda", dtype=torch.float16):
                piece = net(noisy[:, :, r0 : r0 + tile, c0 : c0 + tile], prompt).float()
            out[:, :, r0 : r0 + tile, c0 : c0 + tile] += piece
            weight[:, :, r0 : r0 + tile, c0 : c0 + tile] += 1
    return out / weight


def crop64(cube):
    _, rows, columns = cube.shape
    dr, dc = rows % 64, columns % 64
    return cube[:, dr // 2 : rows - dr + dr // 2, dc // 2 : columns - dc + dc // 2]


def evaluate_synthetic(net, root, tag, device, writer):
    for set_dir in sorted((root / "test").iterdir()):
        for path in sorted(set_dir.glob("*.mat")):
            clean = (
                torch.from_numpy(crop64(hsi_data.load_cube(path)))
                .unsqueeze(0)
                .to(device)
            )
            assert clean.shape[1] == mphsir.BANDS, (path, clean.shape)
            for condition, (prompt_id, apply) in noise.EVAL_CONDITIONS.items():
                seed = zlib.crc32(f"{set_dir.name}/{path.stem}/{condition}".encode())
                generator = torch.Generator(device=device).manual_seed(seed)
                noisy = apply(clean.clone(), generator)
                restored = restore(
                    net, noisy, torch.tensor([prompt_id], device=device)
                ).clamp(0, 1)
                target = clean.clamp(0, 1)
                row = {
                    "checkpoint": tag,
                    "set": set_dir.name,
                    "image": path.stem,
                    "condition": condition,
                    "psnr": psnr_per_band(restored, target),
                    "ssim": ssim_per_band(restored, target),
                    "sam": sam_degrees(restored, target),
                    "psnr_noisy": psnr_per_band(noisy.clamp(0, 1), target),
                }
                writer.writerow(row)
                print(
                    f"[eval:{tag}] {set_dir.name}/{path.stem} {condition}: {row['psnr']:.2f} dB",
                    flush=True,
                )


def evaluate_real(net, root, tag, device):
    rows = []
    for path in sorted((root / "real").glob("*.npy")):
        cube = torch.from_numpy(np.load(path)).unsqueeze(0).to(device)
        restored = (
            restore(net, cube, torch.tensor([1], device=device)).clamp(0, 1)
            if net is not None
            else cube
        )
        rows.append(
            {
                "checkpoint": tag,
                "scene": path.stem,
                "stripe_index": stripe_index(restored),
                "noise_sigma": noise_sigma(restored),
                "sam_to_input": sam_degrees(restored, cube),
            }
        )
    return rows


def summarise(csv_path):
    import pandas as pd

    table = pd.read_csv(csv_path)
    by_set = (
        table.groupby(["checkpoint", "set", "condition"])[["psnr", "ssim", "sam"]]
        .mean()
        .reset_index()
    )
    gate = []
    for (set_name, condition), paper in PAPER.items():
        hit = by_set[
            (by_set.set == set_name)
            & (by_set.condition == condition)
            & (by_set.checkpoint == "official")
        ]
        if len(hit):
            ours = float(hit.psnr.iloc[0])
            gate.append(
                {
                    "set": set_name,
                    "condition": condition,
                    "paper": paper,
                    "ours": round(ours, 2),
                    "delta": round(ours - paper, 2),
                }
            )
    return by_set, gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", required=True, help="'official' or a train.py last.pt"
    )
    parser.add_argument("--tag", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    tag = args.tag or (
        "official"
        if args.checkpoint == "official"
        else Path(args.checkpoint).parent.name
    )
    official = args.data / "Remote_sensing.ckpt"
    net = mphsir.build(official, device)
    if args.checkpoint != "official":
        net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    net.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "checkpoint",
        "set",
        "image",
        "condition",
        "psnr",
        "ssim",
        "sam",
        "psnr_noisy",
    ]
    new_file = not args.out.exists()
    with open(args.out, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        evaluate_synthetic(net, args.data, tag, device, writer)

    real_path = args.out.with_name("real_noref.jsonl")
    with open(real_path, "a") as handle:
        rows = evaluate_real(net, args.data, tag, device)
        if tag == "official":
            rows = evaluate_real(None, args.data, "raw_input", device) + rows
        handle.writelines(json.dumps(row) + "\n" for row in rows)

    by_set, gate = summarise(args.out)
    print(by_set[by_set.checkpoint == tag].to_string(index=False), flush=True)
    if gate:
        print("[gate] Stage 0 reproduction vs paper (PSNR dB):", flush=True)
        for g in gate:
            print(f"  {g}", flush=True)
        args.out.with_name("stage0_gate.json").write_text(json.dumps(gate, indent=1))


if __name__ == "__main__":
    main()
