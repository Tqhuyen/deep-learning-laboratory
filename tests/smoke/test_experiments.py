"""Smoke tests: every experiment must run end to end at the ``tiny`` tier on CPU.

This is the guard that keeps the repository usable on a laptop: if a change
makes an experiment need more than a few seconds of CPU, CI fails here.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = sorted((REPOSITORY_ROOT / "experiments").rglob("run.py"))
EXPERIMENT_IDS = [str(path.relative_to(REPOSITORY_ROOT / "experiments").parent) for path in EXPERIMENTS]


def test_experiments_are_discovered():
    assert len(EXPERIMENTS) >= 3


@pytest.mark.parametrize("script", EXPERIMENTS, ids=EXPERIMENT_IDS)
def test_experiment_runs_at_the_tiny_tier(script: Path, tmp_path: Path):
    environment = {**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT)}
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--tier",
            "tiny",
            "--device",
            "cpu",
            "--output-directory",
            str(tmp_path),
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"{script}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    relative = script.relative_to(REPOSITORY_ROOT / "experiments")
    area, name = relative.parts[0], relative.parts[1]
    run_directory = tmp_path / area / name / "tiny"
    single = (run_directory / "checkpoint.pt").exists()
    per_architecture = any((run_directory / "checkpoints").glob("*.pt"))
    assert single or per_architecture
    assert (run_directory / "config.json").exists()
    assert (run_directory / "metrics.json").exists()
