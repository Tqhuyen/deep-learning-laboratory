"""Statistical tests and confidence intervals for comparison studies.

These are the tools that turn "model A scored 0.3 points higher" into a claim:
paired bootstrap over test examples and an exact McNemar test over predictions.
Multiple pairwise comparisons are corrected with Holm's method.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np


def bootstrap_confidence_interval(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    repetitions: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for ``statistic(values)``."""
    sample = np.asarray(values, dtype=float)
    if sample.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions)
    for index in range(repetitions):
        draw = rng.integers(0, sample.size, sample.size)
        estimates[index] = statistic(sample[draw])
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1.0 - alpha)),
    )


def paired_bootstrap_difference(
    first: Sequence[float],
    second: Sequence[float],
    confidence: float = 0.95,
    repetitions: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap interval for ``mean(first - second)`` over paired examples."""
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired samples must match: {a.shape} vs {b.shape}")
    rng = np.random.default_rng(seed)
    differences = np.empty(repetitions)
    for index in range(repetitions):
        draw = rng.integers(0, a.size, a.size)
        differences[index] = (a[draw] - b[draw]).mean()
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(differences, alpha)),
        float(np.quantile(differences, 1.0 - alpha)),
    )


def mcnemar_exact(correct_first: Sequence[bool], correct_second: Sequence[bool]) -> float:
    """Two-sided exact McNemar p-value for paired binary correctness arrays."""
    first = np.asarray(correct_first, dtype=bool)
    second = np.asarray(correct_second, dtype=bool)
    if first.shape != second.shape:
        raise ValueError(f"paired predictions must match: {first.shape} vs {second.shape}")
    only_first = int(np.sum(first & ~second))
    only_second = int(np.sum(~first & second))
    trials = only_first + only_second
    if trials == 0:
        return 1.0
    smaller = min(only_first, only_second)
    tail = sum(math.comb(trials, k) for k in range(smaller + 1)) / 2**trials
    return float(min(1.0, 2.0 * tail))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down correction, preserving input order."""
    values = list(p_values)
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [1.0] * len(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, values[index] * (total - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_bootstrap_p_value(
    first: Sequence[float],
    second: Sequence[float],
    repetitions: int = 2000,
    seed: int = 0,
) -> float:
    """Two-sided bootstrap p-value for ``mean(first - second) = 0``."""
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired samples must match: {a.shape} vs {b.shape}")
    rng = np.random.default_rng(seed)
    differences = np.empty(repetitions)
    for index in range(repetitions):
        draw = rng.integers(0, a.size, a.size)
        differences[index] = (a[draw] - b[draw]).mean()
    below = float(np.mean(differences <= 0.0))
    above = float(np.mean(differences >= 0.0))
    return float(min(1.0, 2.0 * min(below, above)))
