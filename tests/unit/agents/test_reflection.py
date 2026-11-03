from __future__ import annotations

import pytest

from laboratory.agents.loops import DirectAgent
from laboratory.agents.reflection import ReflexionAgent, read_verdict
from laboratory.agents.scripted import ScriptedModel


@pytest.mark.parametrize(
    ("text", "correct"),
    [
        ("Verdict: CORRECT\nReason: fine", True),
        ("Verdict: INCORRECT\nReason: off by one", False),
        ("**Verdict:** correct", True),
        ("I think the answer is not correct.", False),
        ("The answer looks correct to me.", True),
        ("Unsure.", False),
    ],
)
def test_verdicts(text, correct):
    assert read_verdict(text) is correct


def test_a_failed_trial_becomes_a_lesson_for_the_next_one():
    model = ScriptedModel(
        [
            "Final Answer: 400",
            "Verdict: INCORRECT\nReason: the product is wrong.",
            "I multiplied in my head. Next time use exact arithmetic.",
            "Final Answer: 391",
            "Verdict: CORRECT\nReason: verified.",
        ]
    )
    episode = ReflexionAgent(model, DirectAgent(model), trials=3).run("What is 17 * 23?")
    assert (episode.answer, episode.calls) == ("391", 5)
    assert "Lessons from earlier failed attempts:\n- I multiplied in my head." in model.requests[3]["messages"][1].content
    assert [event["correct"] for event in episode.events] == [False, True]


def test_the_last_trial_is_kept_when_none_passes():
    model = ScriptedModel(["Final Answer: 1", "Verdict: INCORRECT", "lesson", "Final Answer: 2", "Verdict: INCORRECT"])
    episode = ReflexionAgent(model, DirectAgent(model), trials=2).run("q")
    assert episode.answer == "2" and model.remaining == 0


def test_an_external_evaluator_replaces_the_judge_call():
    model = ScriptedModel(["Final Answer: 391"])
    seen = []
    agent = ReflexionAgent(model, DirectAgent(model), evaluator=lambda question, attempt: seen.append(attempt.answer) or (True, "tests pass"))
    assert agent.run("17 * 23?").answer == "391"
    assert seen == ["391"] and model.remaining == 0
