from __future__ import annotations

import re

import pytest

from laboratory.agents.tasks import answers_match, load_task_suite


@pytest.mark.parametrize(
    ("prediction", "reference", "match"),
    [
        ("391", "391", True),
        ("391 crates", "391", True),
        ("17 x 23 = 391", "391", True),
        ("approximately 52786.36", "52786", True),
        ("2,147,483,647", "2147483647", True),
        ("392", "391", False),
        ("I could not compute it", "391", False),
        (None, "391", False),
        ("Brask", "Brask", True),
        ("the town of Brask.", "Brask", True),
        ("Tasso River", "Tasso", True),
        ("Halvern", "Ostmere", False),
        ("Brask is a mountain town in Halvern founded in 1532 with a railway to the observatory", "Brask", False),
    ],
)
def test_answer_matching(prediction, reference, match):
    assert answers_match(prediction, reference) is match


def test_tier_prefixes_cover_every_task_kind():
    """Presets take the first N tasks; the suite interleaves kinds so a 4-task tier sees all four."""
    suite = load_task_suite()
    assert len({task.kind for task in suite.tasks[:4]}) == 4
    assert len({task.identifier for task in suite.tasks}) == len(suite.tasks)


def test_lookup_answers_are_stated_in_the_encyclopedia():
    suite = load_task_suite()
    text = " ".join(suite.knowledge.values()).replace(",", "")
    for task in suite.tasks:
        if task.kind in ("lookup", "multi_hop"):
            assert re.search(rf"\b{re.escape(task.answer)}\b", text), task.identifier
