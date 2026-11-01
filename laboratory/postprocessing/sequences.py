"""Postprocessing: model ids to task-usable output.

Inference produces token ids. Postprocessing removes control tokens and turns
the remainder into the unit the task is scored in (a string, a label, a box).
"""

from __future__ import annotations

from typing import Iterable


def trim_at_eos(token_ids: Iterable[int], eos_id: int) -> list[int]:
    """Keep ids before the first ``eos_id`` (exclusive)."""
    trimmed: list[int] = []
    for token_id in token_ids:
        if token_id == eos_id:
            break
        trimmed.append(token_id)
    return trimmed
