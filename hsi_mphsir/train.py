"""Fine-tune the official MP-HSIR remote-sensing checkpoint under a fixed wall-clock budget.

Arms (identical data order, schedule, budget and seed; only the noise differs):
  control  standard noise: half Gaussian sigma U(30, 70), half MP-HSIR complex noise
  physics  D1: 60 % physics sensor noise, 40 % the control mixture

The learning rate follows warmup (5 %) then cosine to ``1e-6`` over the *time* budget, so both arms see the same
schedule even if their step rates differ. L1 loss on clamped output and fp16 autocast follow MP-HSIR ``train.py``.

    python hsi_mphsir/train.py --arm physics --minutes 180 --data /workspace/hsi_data --out runs/physics
"""

import argparse
import json
import math
import time
from pathlib import Path

import model as mphsir
import noise
import numpy as np
import torch


def learning_rate(fraction, peak, floor=1e-6, warmup=0.05):
    if fraction < warmup:
        return peak * max(fraction / warmup, 0.01)
    progress = min((fraction - warmup) / (1 - warmup), 1.0)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))


def augment(batch, generator):
    """Random flips and 90-degree rotations in the spatial plane only; band order is physical and never flipped."""
    k = int(torch.randint(0, 4, (1,), generator=generator))
    batch = torch.rot90(batch, k, dims=(2, 3))
    if bool(torch.randint(0, 2, (1,), generator=generator)):
        batch = batch.flip(3)
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("control", "physics"), required=True)
    parser.add_argument("--minutes", type=float, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--save-every-minutes", type=float, default=30)
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args.out.mkdir(parents=True, exist_ok=True)
    patches = torch.from_numpy(np.load(args.data / "train_patches.npy")).pin_memory()
    net = mphsir.build(args.data / "Remote_sensing.ckpt", device)
    net.train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = torch.nn.L1Loss()
    data_rng = torch.Generator().manual_seed(args.seed)
    noise_rng = torch.Generator(device=device).manual_seed(args.seed)
    mix = noise.standard if args.arm == "control" else noise.physics_mix

    budget = args.minutes * 60
    start = last_save = time.time()
    step, running = 0, []
    with open(args.out / "train_log.jsonl", "a") as log:
        while (elapsed := time.time() - start) < budget:
            lr = learning_rate(elapsed / budget, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            index = torch.randint(0, len(patches), (args.batch,), generator=data_rng)
            clean = augment(
                patches[index].to(device, non_blocking=True).float(), data_rng
            )
            with torch.no_grad():
                noisy, prompt = mix(clean, noise_rng)
            with torch.autocast("cuda", dtype=torch.float16):
                restored = net(noisy, prompt)
            loss = loss_fn(restored.float().clamp(0, 1), clean)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            running.append(loss.item())
            if step % 50 == 0:
                row = {
                    "step": step,
                    "minutes": round(elapsed / 60, 2),
                    "lr": lr,
                    "loss": float(np.mean(running)),
                    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                }
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(f"[train:{args.arm}] {row}", flush=True)
                running.clear()
            if time.time() - last_save > args.save_every_minutes * 60:
                torch.save(net.state_dict(), args.out / "last.pt")
                last_save = time.time()
    torch.save(net.state_dict(), args.out / "last.pt")
    summary = {
        "arm": args.arm,
        "steps": step,
        "minutes": round((time.time() - start) / 60, 1),
        "batch": args.batch,
        "patches_seen": step * args.batch,
        "lr_peak": args.lr,
        "seed": args.seed,
    }
    (args.out / "train_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[train:{args.arm}] done {summary}", flush=True)


if __name__ == "__main__":
    main()
