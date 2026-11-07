"""Muon: momentum orthogonalised by Newton–Schulz, for hidden weight matrices.

Muon (Jordan et al., 2024) replaces the update of each weight matrix with the
nearest semi-orthogonal matrix to its Nesterov momentum, computed by five
quintic Newton–Schulz iterations; convolution kernels are flattened to
``(out, in · kh · kw)``. It is meant for hidden matrices only, so embeddings,
the output layer, biases and normalisation gains are updated by AdamW inside
the same optimizer, as in the reference implementation.

Muon updates are scaled by ``0.2 · sqrt(max(rows, columns))`` (Liu et al.,
2025, "Muon is Scalable for LLM Training"), which matches their RMS to AdamW's:
learning rates and weight decay tuned for AdamW carry over unchanged.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn

NEWTON_SCHULZ = (3.4445, -4.7750, 2.0315)


def newton_schulz(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the orthogonal factor ``U V^T`` of ``matrix = U S V^T``; singular values end up in about [0.7, 1.2]."""
    a, b, c = NEWTON_SCHULZ
    native_bf16 = matrix.is_cuda and torch.cuda.get_device_capability(matrix.device)[0] >= 8
    x = matrix.to(torch.bfloat16 if native_bf16 else torch.float32)
    tall = x.size(-2) > x.size(-1)
    if tall:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    return (x.mT if tall else x).to(matrix.dtype)


def muon_parameter_groups(model: nn.Module, adamw_modules: Iterable[str] = ("head", "lm_head")) -> list[dict]:
    """Split ``model``'s parameters into a Muon group and an AdamW group.

    AdamW gets every parameter with fewer than two dimensions, every
    ``nn.Embedding`` weight (with any output layer tied to it) and every
    parameter of a module named in ``adamw_modules``, matched against the
    qualified name or its last component. Muon gets the remaining matrices.
    """
    names = set(adamw_modules)
    adamw = {
        id(p)
        for name, module in model.named_modules()
        if isinstance(module, nn.Embedding) or name in names or name.rsplit(".", 1)[-1] in names
        for p in module.parameters(recurse=False)
    }
    muon, other = [], []
    for p in model.parameters():  # tied weights appear once
        (other if p.ndim < 2 or id(p) in adamw else muon).append(p)
    return [{"params": muon, "use_muon": True}, {"params": other, "use_muon": False}]


class Muon(torch.optim.Optimizer):
    """Muon for hidden weight matrices, AdamW for every other parameter.

    ``params`` holds tensors, split by shape (two or more dimensions go to
    Muon), or groups with a ``use_muon`` flag such as
    :func:`muon_parameter_groups` returns. ``lr`` and decoupled
    ``weight_decay`` apply to both kinds; ``momentum``, ``nesterov`` and
    ``ns_steps`` to Muon; ``betas`` and ``eps`` to AdamW.
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        params = list(params)
        if params and not isinstance(params[0], dict):
            params = [
                {"params": [p for p in params if p.ndim >= 2], "use_muon": True},
                {"params": [p for p in params if p.ndim < 2], "use_muon": False},
            ]
        groups = [{**group, "params": list(group["params"])} for group in params]
        if any("use_muon" not in group for group in groups):
            raise ValueError("every parameter group needs a 'use_muon' flag")
        defaults = dict(
            lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
            ns_steps=ns_steps, betas=betas, eps=eps,
        )
        super().__init__([group for group in groups if group["params"]], defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            update = self._muon_update if group["use_muon"] else self._adamw_update
            for p in group["params"]:
                if p.grad is None:
                    continue
                direction = update(p.grad, self.state[p], group)
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(direction, alpha=-group["lr"])
        return loss

    @staticmethod
    def _muon_update(grad, state, group):
        if not state:
            state["momentum_buffer"] = torch.zeros_like(grad)
        buffer = state["momentum_buffer"]
        buffer.lerp_(grad, 1 - group["momentum"])
        momentum = grad.lerp(buffer, group["momentum"]) if group["nesterov"] else buffer
        matrix = newton_schulz(momentum.reshape(len(momentum), -1), group["ns_steps"])
        return matrix.reshape_as(grad) * (0.2 * math.sqrt(max(matrix.shape)))

    @staticmethod
    def _adamw_update(grad, state, group):
        beta1, beta2 = group["betas"]
        if not state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(grad)
            state["exp_avg_sq"] = torch.zeros_like(grad)
        state["step"] += 1
        state["exp_avg"].lerp_(grad, 1 - beta1)
        state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        corrected = (state["exp_avg_sq"] / (1 - beta2 ** state["step"])).sqrt().add_(group["eps"])
        return state["exp_avg"] / (1 - beta1 ** state["step"]) / corrected
