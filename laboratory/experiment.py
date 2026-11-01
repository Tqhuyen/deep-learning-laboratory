"""Conventions shared by every experiment in the repository.

An **experiment** is a directory under ``experiments/<area>/<name>/`` holding a
``run.py`` that complies with this contract:

1. It defines ``PRESETS = {"tiny": ..., "smoke": ..., "development": ...,
   "full": ...}``.
2. It obtains arguments with :func:`argument_parser`, which guarantees
   ``--tier``, ``--seed``, ``--device`` and ``--output-directory``.
3. It writes its resolved config and artifacts under
   ``runs/<area>/<name>/<tier>/`` via :func:`prepare_run` and
   :func:`write_config`.
4. It needs no network access and creates its own data if the paper's dataset is
   not in ``data/``.

Tiers describe *how much compute* a run may spend, not what the model is:

=============  ==========================  ==================================
tier           budget                      purpose
=============  ==========================  ==================================
``tiny``       seconds on CPU              plumbing: shapes, masks, one step
``smoke``      about a minute on CPU       learning signal: loss must fall
``development`` minutes on one GPU          real experiments while iterating
``full``       as much as you have         reported numbers
=============  ==========================  ==================================

``tiny`` must fit in a few seconds on a laptop; the test suite asserts that by
running every experiment at that tier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from .utilities import get_device

TIERS = ("tiny", "smoke", "development", "full")


def repository_root() -> Path:
    """Path to the repository root (the directory containing ``laboratory/``)."""
    return Path(__file__).resolve().parents[1]


def data_path(filename: str) -> Path:
    """Resolve a file under ``data/`` and fail with a readable message."""
    path = repository_root() / "data" / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Download the corpus into data/ or change the experiment's data source."
        )
    return path


def argument_parser(description: str) -> argparse.ArgumentParser:
    """Parser with the arguments every experiment must accept."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--tier",
        choices=TIERS,
        default="tiny",
        help="compute budget: tiny, smoke, development or full",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for python, numpy and torch")
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu', 'cuda', or a specific device such as 'cuda:1'",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="override the preset's optimizer-step budget (useful for debugging)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("runs"),
        help="where to write config, checkpoints and samples",
    )
    return parser


def select_preset(presets: Mapping[str, Mapping], tier: str) -> dict:
    """Return a mutable copy of ``presets[tier]`` with a clear error if absent."""
    if tier not in presets:
        raise SystemExit(
            f"tier {tier!r} is not implemented here; available tiers: {sorted(presets)}"
        )
    return dict(presets[tier])


def prepare_run(area: str, name: str, tier: str, output_directory: Path) -> Path:
    """Create (if needed) and return ``<output>/<area>/<name>/<tier>``."""
    run_directory = Path(output_directory) / area / name / tier
    run_directory.mkdir(parents=True, exist_ok=True)
    return run_directory


def write_config(run_directory: Path, config: Mapping) -> None:
    """Persist the resolved configuration next to the run's artifacts."""
    (Path(run_directory) / "config.json").write_text(
        json.dumps(dict(config), indent=2, default=str), encoding="utf-8"
    )


def write_metrics(run_directory: Path, history: list[dict], **extra) -> None:
    """Persist training history and final metrics as JSON."""
    payload = {"history": history, **extra}
    (Path(run_directory) / "metrics.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


def environment_summary(device: str | None = None) -> str:
    """One-line description of the compute and library versions in use."""
    import platform

    import torch

    chosen = get_device(device or "auto")
    gpu = torch.cuda.get_device_name(chosen) if chosen.type == "cuda" else "no CUDA"
    return (
        f"device={chosen} ({gpu}) torch={torch.__version__} "
        f"python={platform.python_version()} threads={torch.get_num_threads()}"
    )
