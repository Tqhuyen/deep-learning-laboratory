from __future__ import annotations

import numpy as np

from laboratory.evaluation.statistics import (
    bootstrap_confidence_interval,
    holm_adjust,
    mcnemar_exact,
    paired_bootstrap_difference,
    paired_bootstrap_p_value,
)


def test_bootstrap_interval_brackets_the_mean():
    values = np.arange(100, dtype=float)
    low, high = bootstrap_confidence_interval(values, repetitions=500, seed=1)
    assert low <= values.mean() <= high
    assert low < high


def test_paired_bootstrap_of_a_constant_offset_is_exact():
    first = np.ones(50)
    second = np.zeros(50)
    low, high = paired_bootstrap_difference(first, second, repetitions=200, seed=0)
    assert abs(low - 1.0) < 1e-12 and abs(high - 1.0) < 1e-12


def test_bootstrap_p_value_is_one_for_identical_and_zero_for_separated_samples():
    same = np.arange(20, dtype=float)
    assert paired_bootstrap_p_value(same, same, repetitions=300) == 1.0
    assert paired_bootstrap_p_value(same + 1.0, same, repetitions=300) == 0.0


def test_mcnemar_uses_only_discordant_pairs():
    first = np.array([True, True, False, False])
    second = np.array([True, False, True, False])
    assert mcnemar_exact(first, second) == 1.0

    first = np.array([True, True, True, False])
    second = np.array([False, False, False, False])
    assert abs(mcnemar_exact(first, second) - 0.25) < 1e-12


def test_holm_adjust_matches_the_step_down_definition():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])
