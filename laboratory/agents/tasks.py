"""Task suites and answer scoring for agent experiments.

``data/agent_tasks.json`` holds a small encyclopedia about a **fictional**
world plus questions in four kinds:

* ``arithmetic`` — exact multi-digit arithmetic (a calculator beats mental maths);
* ``lookup`` — one encyclopedia lookup;
* ``multi_hop`` — chained lookups (A's designer -> designer's birthplace);
* ``lookup_arithmetic`` — lookups followed by a calculation.

Because the world is invented, no model can answer from memorised training
data: correct answers on the lookup kinds come from using the tool, which is
exactly the behaviour architectures differ in.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .loops import canonical_answer, numbers_in
from .tools import Toolbox, calculator, lookup_tool

_NUMERIC_REFERENCE = re.compile(r"[-+]?\$?[\d,]*\.?\d+%?")


@dataclass(frozen=True)
class Task:
    identifier: str
    kind: str
    question: str
    answer: str


@dataclass(frozen=True)
class TaskSuite:
    tasks: tuple[Task, ...]
    knowledge: Mapping[str, str]

    def toolbox(self) -> Toolbox:
        """``calculator`` plus ``lookup`` over this suite's encyclopedia."""
        return Toolbox([calculator(), lookup_tool(self.knowledge)])

    def of_kind(self, kind: str) -> list[Task]:
        return [task for task in self.tasks if task.kind == kind]


def task_suite_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "agent_tasks.json"


def load_task_suite(path: Path | str | None = None) -> TaskSuite:
    payload = json.loads(Path(path or task_suite_path()).read_text(encoding="utf-8"))
    tasks = tuple(
        Task(row["id"], row["kind"], row["question"], str(row["answer"])) for row in payload["tasks"]
    )
    return TaskSuite(tasks, dict(payload["knowledge"]))


def answers_match(prediction: str | None, reference: str) -> bool:
    """Lenient but unambiguous answer matching.

    Numeric references match when the first or last number in the prediction is
    within a relative tolerance of 1e-4 (``"391 crates"`` and ``"17 x 23 = 391"``
    both count). Text references match on canonical words, or when the
    reference phrase appears inside a short prediction (``"Brask"`` matches
    ``"the town of Brask"`` but not a paragraph that mentions Brask).
    """
    if prediction is None:
        return False
    if _NUMERIC_REFERENCE.fullmatch(reference.strip()):
        target = numbers_in(reference)[0]
        found = numbers_in(prediction)
        return bool(found) and any(
            math.isclose(value, target, rel_tol=1e-4, abs_tol=1e-9) for value in (found[0], found[-1])
        )
    expected, predicted = canonical_answer(reference), canonical_answer(prediction)
    if not expected or not predicted:
        return False
    if expected == predicted:
        return True
    return f" {expected} " in f" {predicted} " and len(predicted.split()) <= len(expected.split()) + 6
