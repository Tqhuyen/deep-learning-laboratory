"""Plan-and-execute: decide the steps first, then carry them out one at a time.

Wang et al., *Plan-and-Solve Prompting* (https://arxiv.org/abs/2305.04091),
in the agentic form popularised by BabyAGI and LangChain's plan-and-execute:

1. a **planner** call turns the question into a short numbered plan;
2. an **executor** agent (ReAct or tool calling, with tools) solves each step,
   seeing the plan and the results so far;
3. a **solver** call composes the final answer from the step results.

The question it answers in a comparison: does separating *what to do* from
*doing it* help multi-hop tasks, and what does the extra planning call cost?
"""

from __future__ import annotations

import re

from .loops import Agent, Episode, extract_final_answer, last_line
from .messages import ChatModel, Message
from .tools import Toolbox

PLANNER_PROMPT = """You are the planner. Break the question into at most {max_steps} short, concrete steps that can be carried out one at a time with these tools:
{tools}

Reply with a numbered list of steps only. Do not carry the steps out."""

STEP_PROMPT = """Overall question: {question}

Plan:
{plan}

Results so far:
{results}

Current step ({index} of {count}): {step}
Carry out only the current step and give its result as the Final Answer."""

SOLVER_PROMPT = (
    "Answer the question from the results of the plan's steps. Finish with a final line of the form\n"
    "Final Answer: <the answer only>"
)


def parse_plan(text: str) -> list[str]:
    """Steps of a numbered or bulleted list (``1.``, ``2)``, ``-``, ``Step 3:``)."""
    steps = []
    for line in text.splitlines():
        match = re.match(r"\s*(?:\d+\s*[.)]|[-*•]|step\s*\d+\s*[:.)])\s*(.+)", line, re.IGNORECASE)
        if match and match.group(1).strip():
            steps.append(match.group(1).strip().strip("*").strip())
    return steps


class PlanAndExecuteAgent(Agent):
    name = "plan_and_execute"

    def __init__(self, model: ChatModel, executor: Agent, toolbox: Toolbox, *, max_plan_steps: int = 5, **sampling):
        super().__init__(model, **sampling)
        self.executor = executor
        self.toolbox = toolbox
        self.max_plan_steps = max_plan_steps
        self.requires_tools = executor.requires_tools

    def solve(self, episode: Episode) -> None:
        planner = [
            Message.system(PLANNER_PROMPT.format(max_steps=self.max_plan_steps, tools=self.toolbox.describe())),
            Message.user(episode.question),
        ]
        plan_completion = self.call(episode, planner)
        steps = parse_plan(plan_completion.text)[: self.max_plan_steps] or [episode.question]
        plan = "\n".join(f"{index}. {step}" for index, step in enumerate(steps, start=1))
        episode.events.append({"plan": steps})

        results: list[str] = []
        for index, step in enumerate(steps, start=1):
            sub_episode = self.executor.run(
                STEP_PROMPT.format(
                    question=episode.question,
                    plan=plan,
                    results="\n".join(results) or "(none yet)",
                    index=index,
                    count=len(steps),
                    step=step,
                )
            )
            episode.absorb(sub_episode)
            outcome = sub_episode.answer if sub_episode.answer is not None else f"(no result: {sub_episode.stopped})"
            results.append(f"{index}. {step} -> {outcome}")
            episode.events.append({"step": step, "result": sub_episode.answer, "stopped": sub_episode.stopped})

        solver = [
            Message.system(SOLVER_PROMPT),
            Message.user(f"Question: {episode.question}\n\nStep results:\n" + "\n".join(results)),
        ]
        final = self.call(episode, solver)
        episode.messages = [*planner, plan_completion.message, *solver, final.message]
        episode.answer = extract_final_answer(final.text) or last_line(final.text)
