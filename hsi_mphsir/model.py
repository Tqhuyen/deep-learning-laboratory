"""Load the official MP-HSIR remote-sensing network (100 bands, dim 96, 7 task prompts).

The MP-HSIR repository must be cloned and its root given by ``MPHSIR_DIR`` (pinned commit in ``run_vast.sh``).
Lightning checkpoints store weights under ``net.``; fine-tuned checkpoints from ``train.py`` store the bare state.
"""

import os
import sys

import torch

BANDS, DIM, TASKS = 100, 96, 7


def build(checkpoint, device):
    sys.path.insert(0, os.environ["MPHSIR_DIR"])
    from net.MP_HSIR import MP_HSIR_Net

    net = MP_HSIR_Net(in_channel=BANDS, out_channel=BANDS, dim=DIM, task_classes=TASKS)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    state = {k.removeprefix("net."): v for k, v in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if "clip" not in k]
    if real_missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={real_missing[:5]} unexpected={unexpected[:5]}"
        )
    return net.to(device)
