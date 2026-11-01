from __future__ import annotations

from laboratory.postprocessing.sequences import trim_at_eos


def test_trim_at_eos_keeps_only_tokens_before_eos():
    assert trim_at_eos([5, 6, 2, 9], eos_id=2) == [5, 6]


def test_trim_at_eos_returns_everything_without_eos():
    assert trim_at_eos([5, 6], eos_id=2) == [5, 6]


def test_trim_at_eos_handles_an_immediate_eos():
    assert trim_at_eos([2, 5], eos_id=2) == []
