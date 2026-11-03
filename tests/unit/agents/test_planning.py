from __future__ import annotations

import pytest

from laboratory.agents.loops import ReActAgent
from laboratory.agents.planning import PlanAndExecuteAgent, parse_plan
from laboratory.agents.scripted import ScriptedModel
from laboratory.agents.tools import Toolbox, calculator, lookup_tool

TOOLBOX = Toolbox([calculator(), lookup_tool({"Velmora Bridge": "Designed by Ilse Marquant.", "Ilse Marquant": "Born in Brask."})])


@pytest.mark.parametrize(
    ("text", "steps"),
    [
        ("1. Find the designer.\n2) Find the birthplace.", ["Find the designer.", "Find the birthplace."]),
        ("Plan:\n- look up A\n* look up B\n", ["look up A", "look up B"]),
        ("Step 1: do x\nStep 2: do y", ["do x", "do y"]),
        ("No list at all.", []),
    ],
)
def test_plans_are_read_from_numbered_or_bulleted_lists(text, steps):
    assert parse_plan(text) == steps


def test_executes_each_step_with_the_results_so_far_then_solves():
    model = ScriptedModel(
        [
            "1. Look up the designer of the Velmora Bridge.\n2. Look up where the designer was born.",
            "Action: lookup\nAction Input: Velmora Bridge",
            "Final Answer: Ilse Marquant",
            "Action: lookup\nAction Input: Ilse Marquant",
            "Final Answer: Brask",
            "Final Answer: Brask",
        ]
    )
    executor = ReActAgent(model, TOOLBOX)
    episode = PlanAndExecuteAgent(model, executor, TOOLBOX).run("Where was the designer of the Velmora Bridge born?")
    assert (episode.answer, episode.calls, episode.tool_calls) == ("Brask", 6, 2)
    second_step = model.requests[3]["messages"][1].content
    assert "Current step (2 of 2)" in second_step and "-> Ilse Marquant" in second_step
    solver = model.requests[5]["messages"][1].content
    assert "1. Look up the designer of the Velmora Bridge. -> Ilse Marquant" in solver
    assert episode.events[0] == {"plan": ["Look up the designer of the Velmora Bridge.", "Look up where the designer was born."]}


def test_plan_length_is_capped():
    model = ScriptedModel(["1. a\n2. b\n3. c", "Final Answer: x", "Final Answer: y"])
    episode = PlanAndExecuteAgent(model, ReActAgent(model, TOOLBOX), TOOLBOX, max_plan_steps=1).run("q")
    assert episode.events[0]["plan"] == ["a"] and episode.answer == "y"
