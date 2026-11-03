from __future__ import annotations

import json

import pytest

from laboratory.agents.loops import (
    DirectAgent,
    ReActAgent,
    ToolCallingAgent,
    canonical_answer,
    extract_final_answer,
    parse_action,
    truncate_at_observation,
)
from laboratory.agents.messages import Completion, Message, ToolCall, Usage
from laboratory.agents.scripted import ScriptedModel, tool_call_message
from laboratory.agents.tools import Toolbox, calculator, lookup_tool

TOOLBOX = Toolbox([calculator(), lookup_tool({"Velmora Bridge": "Completed in 1911."})])


@pytest.mark.parametrize(
    ("text", "answer"),
    [
        ("Thought: done\nFinal Answer: 391", "391"),
        ("**Final Answer:** Brask\n", "Brask"),
        ("Final Answer: wrong\nWait.\nFinal Answer: $\\boxed{42}$", "42"),
        ("final answer:\n\n  Teodor Vaskel  ", "Teodor Vaskel"),
        ("The answer is 391.", None),
        ("Final Answer:   ", None),
    ],
)
def test_final_answer_extraction(text, answer):
    assert extract_final_answer(text) == answer


@pytest.mark.parametrize(
    ("text", "canonical"),
    [("$1,234.50", "1234.5"), ("391 crates.", "391"), ("The town of Brask.", "town of brask"), (None, None)],
)
def test_canonical_answers(text, canonical):
    assert canonical_answer(text) == canonical


@pytest.mark.parametrize(
    ("text", "tool", "arguments"),
    [
        ('Thought: x\nAction: calculator\nAction Input: {"expression": "17*23"}\n', "calculator", '{"expression": "17*23"}'),
        ("Action: lookup[Velmora Bridge]", "lookup", "Velmora Bridge"),
        ("**Action:** `calculator`\n**Action Input:** 17 * 23", "calculator", "17 * 23"),
        ('Action: calculator\nAction Input: ```json\n{"expression": "1+1"}\n```', "calculator", '{"expression": "1+1"}'),
    ],
)
def test_action_parsing_accepts_common_formats(text, tool, arguments):
    action = parse_action(text)
    assert (action.tool, action.arguments) == (tool, arguments)


def test_no_action_is_none_and_hallucinated_observations_are_cut():
    assert parse_action("Thought: I am thinking.") is None
    assert truncate_at_observation("Action: calculator\nAction Input: 1+1\nObservation: 3\nFinal Answer: 3") == (
        "Action: calculator\nAction Input: 1+1"
    )


def test_react_runs_the_tool_and_feeds_the_observation_back():
    model = ScriptedModel(["Thought: multiply\nAction: calculator\nAction Input: 17 * 23", "Final Answer: 391"])
    episode = ReActAgent(model, TOOLBOX).run("What is 17 * 23?")
    assert (episode.answer, episode.stopped, episode.tool_calls, episode.calls) == ("391", "answer", 1, 2)
    assert model.requests[1]["messages"][-1].content == "Observation: 391"
    assert model.requests[0]["stop"] == ["\nObservation:"]
    assert "calculator(expression: string)" in model.requests[0]["messages"][0].content


def test_react_executes_the_action_instead_of_trusting_a_self_written_observation():
    model = ScriptedModel(["Action: calculator\nAction Input: 17 * 23\nObservation: 400\nFinal Answer: 400", "Final Answer: 391"])
    episode = ReActAgent(model, TOOLBOX).run("What is 17 * 23?")
    assert episode.answer == "391"
    assert model.requests[1]["messages"][-1].content == "Observation: 391"


def test_react_reminds_the_format_and_stops_at_the_step_budget():
    model = ScriptedModel(["I think it is 391."] * 3)
    episode = ReActAgent(model, TOOLBOX, max_steps=3).run("What is 17 * 23?")
    assert (episode.answer, episode.stopped, episode.calls) == (None, "max_steps", 3)
    assert model.requests[1]["messages"][-1].content.startswith("Observation: I could not find an Action")


def test_tool_calling_answers_every_call_with_a_tool_message():
    calls = Message.assistant("", [ToolCall("a", "calculator", '{"expression": "1+1"}'), ToolCall("b", "lookup", '{"title": "velmora bridge"}')])
    model = ScriptedModel([calls, "Final Answer: 1911"])
    episode = ToolCallingAgent(model, TOOLBOX).run("When?")
    assert (episode.answer, episode.tool_calls) == ("1911", 2)
    tool_messages = [m for m in model.requests[1]["messages"] if m.role == "tool"]
    assert [(m.tool_call_id, m.content) for m in tool_messages] == [("a", "2"), ("b", "Completed in 1911.")]
    assert model.requests[0]["tools"] == ["calculator", "lookup"]


def test_tool_calling_takes_the_last_line_when_there_is_no_marker():
    model = ScriptedModel([tool_call_message("calculator", {"expression": "6*7"}), "Let me see.\nIt is 42."])
    assert ToolCallingAgent(model, TOOLBOX).run("6*7?").answer == "It is 42."


def test_direct_agent_uses_no_tools():
    model = ScriptedModel(["Final Answer: Brask"])
    assert DirectAgent(model).run("Where?").answer == "Brask"
    assert model.requests[0]["tools"] == []


def test_model_failures_end_the_episode_instead_of_raising():
    episode = ReActAgent(ScriptedModel([]), TOOLBOX).run("anything")
    assert episode.stopped == "error" and "script exhausted" in episode.error


def test_episode_records_are_json_and_sum_usage():
    usage_reply = Completion(Message.assistant("Final Answer: 1"), usage=Usage(10, 5), provider="p", model="m", latency=0.5)
    episode = DirectAgent(ScriptedModel([usage_reply])).run("q")
    record = json.loads(json.dumps(episode.to_record()))
    assert (record["prompt_tokens"], record["completion_tokens"], record["latency"]) == (10, 5, 0.5)
    assert record["completions"][0]["endpoint"] == "p:m"
