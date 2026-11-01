"""The training loop and checkpointing.

The loop is about 80 lines of real logic: move the batch to the device, run the
step function, scale the loss, clip, step, schedule, log. Everything
task-specific lives in ``step_fn(model, batch) -> (loss, logs)``, so the same
trainer drives a language model, a translation model, a convolutional network
or an LSTM.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .metrics import RunningMean


def _to_device(batch, device: torch.device):
    if isinstance(batch, dict):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    if isinstance(batch, (tuple, list)):
        return type(batch)(v.to(device) if torch.is_tensor(v) else v for v in batch)
    if torch.is_tensor(batch):
        return batch.to(device)
    return batch


def _batch_size(batch) -> int:
    if isinstance(batch, dict):
        return int(next(iter(batch.values())).size(0))
    if isinstance(batch, (tuple, list)):
        return int(batch[0].size(0))
    return int(batch.size(0))


class Trainer:
    """Task-agnostic training loop.

    Args:
        model: any ``nn.Module``; it is moved to the device for you.
        optimizer: e.g. ``torch.optim.AdamW(model.parameters(), lr=3e-4)``.
        device: ``"auto"`` picks CUDA when available.
        clip_grad: max global gradient norm (0 disables clipping).
        amp: mixed precision, only enabled on CUDA.
        scheduler: stepped once per optimizer step, e.g. from
            :mod:`laboratory.training.schedules`.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str | torch.device = "auto",
        clip_grad: float = 1.0,
        amp: bool = False,
        scheduler=None,
    ):
        if str(device) == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip_grad = clip_grad
        self.amp = bool(amp) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp)
        self.step_count = 0

    def train_step(self, batch, step_fn) -> dict[str, float]:
        """One optimizer step. ``step_fn`` returns ``(loss, metrics_dict)``."""
        self.model.train()
        batch = _to_device(batch, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device.type, enabled=self.amp):
            loss, logs = step_fn(self.model, batch)
        self.scaler.scale(loss).backward()
        if self.clip_grad:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        self.step_count += 1
        return {"loss": loss.item(), **{k: float(v) for k, v in logs.items()}}

    @torch.no_grad()
    def evaluate(self, loader, step_fn, max_batches: int | None = None) -> dict[str, float]:
        self.model.eval()
        loss_mean, metric_means = RunningMean(), {}
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = _to_device(batch, self.device)
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                loss, logs = step_fn(self.model, batch)
            n = _batch_size(batch)
            loss_mean.update(loss.item(), n)
            for key, value in logs.items():
                metric_means.setdefault(key, RunningMean()).update(float(value), n)
        return {"loss": loss_mean.mean, **{k: v.mean for k, v in metric_means.items()}}

    def fit(
        self,
        train_loader,
        step_fn,
        epochs: int = 1,
        val_loader=None,
        log_every: int = 50,
        max_steps: int | None = None,
        validation_batches: int | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> list[dict[str, float]]:
        """Train for ``epochs``, or stop early at ``max_steps`` optimizer steps."""
        history: list[dict[str, float]] = []
        for epoch in range(1, epochs + 1):
            epoch_loss = RunningMean()
            for batch_index, batch in enumerate(train_loader, start=1):
                logs = self.train_step(batch, step_fn)
                epoch_loss.update(logs["loss"])

                if log_every and batch_index % log_every == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    extra = " ".join(f"{k}={v:.4f}" for k, v in logs.items() if k != "loss")
                    print(
                        f"epoch {epoch} step {self.step_count} "
                        f"loss={logs['loss']:.4f} lr={lr:.2e} {extra}".rstrip()
                    )
                if max_steps is not None and self.step_count >= max_steps:
                    break

            record = {"epoch": epoch, "train_loss": epoch_loss.mean}
            if val_loader is not None:
                val = self.evaluate(val_loader, step_fn, max_batches=validation_batches)
                record.update({f"val_{k}": v for k, v in val.items()})
                print(
                    f"epoch {epoch}: train_loss={epoch_loss.mean:.4f} "
                    + " ".join(f"val_{k}={v:.4f}" for k, v in val.items())
                )
            else:
                print(f"epoch {epoch}: train_loss={epoch_loss.mean:.4f}")
            history.append(record)

            if checkpoint_path is not None:
                save_checkpoint(checkpoint_path, self.model, self.optimizer)
            if max_steps is not None and self.step_count >= max_steps:
                break
        return history


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    **extra,
) -> None:
    """Save weights plus optional optimizer state and arbitrary metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), **extra}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load what :func:`save_checkpoint` wrote; returns the metadata."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return {k: v for k, v in payload.items() if k not in ("model", "optimizer")}
