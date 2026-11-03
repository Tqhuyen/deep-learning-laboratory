"""Reflexion: try, judge, write down the lesson, try again.

Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning*
(https://arxiv.org/abs/2303.11366). An **actor** agent attempts the task; an
**evaluator** judges the attempt; on failure a **self-reflection** call turns
the trajectory into a short lesson that is prepended to the next attempt.

The paper's evaluator is an external signal (unit tests, environment reward).
Here the default evaluator is the model itself, which never sees the reference
answer, so the comparison stays honest. Pass ``evaluator=`` to plug in a real
signal (tests for code, a verifier for maths) when your task has one.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from .loops import Agent, Episode
from .messages import ChatModel, Message

EVALUATOR_PROMPT = """You check another agent's work. Given a question, the agent's trajectory and its final answer, decide whether the answer is correct and supported by the trajectory (tool results are evidence). Check arithmetic carefully.

Reply with exactly two lines:
Verdict: CORRECT or INCORRECT
Reason: one sentence"""

REFLECTION_PROMPT = (
    "An attempt to answer the question failed. In two or three sentences, diagnose what went wrong "
    "and state a concrete plan for the next attempt. Do not answer the question."
)

Evaluator = Callable[[str, Episode], "tuple[bool, str]"]


def render_trajectory(episode: Episode, limit: int = 4000) -> str:
    """The non-system turns of an episode as text, keeping the last ``limit`` characters."""
    lines = []
    for message in episode.messages:
        if message.role == "system":
            continue
        calls = "".join(f" [calls {call.name}({call.arguments})]" for call in message.tool_calls)
        lines.append(f"{message.role}: {message.content}{calls}")
    text = "\n".join(lines)
    return text if len(text) <= limit else "..." + text[-limit:]


def read_verdict(text: str) -> bool:
    """True only for an explicit CORRECT; unparseable verdicts count as failures."""
    labelled = re.search(r"verdict\W*(in)?correct", text, re.IGNORECASE)
    if labelled:
        return labelled.group(1) is None
    if re.search(r"\bincorrect\b|\bnot correct\b|\bwrong\b", text, re.IGNORECASE):
        return False
    return bool(re.search(r"\bcorrect\b", text, re.IGNORECASE))


class ReflexionAgent(Agent):
    name = "reflexion"

    def __init__(
        self,
        model: ChatModel,
        actor: Agent,
        *,
        trials: int = 3,
        evaluator: Evaluator | None = None,
        **sampling,
    ):
        super().__init__(model, **sampling)
        if trials < 1:
            raise ValueError("Reflexion needs at least one trial")
        self.actor = actor
        self.trials = trials
        self.evaluator = evaluator
        self.requires_tools = actor.requires_tools

    def evaluate(self, episode: Episode, attempt: Episode) -> tuple[bool, str]:
        if attempt.answer is None:
            return False, f"the attempt ended without an answer ({attempt.stopped})"
        if self.evaluator is not None:
            return self.evaluator(episode.question, attempt)
        completion = self.call(
            episode,
            [
                Message.system(EVALUATOR_PROMPT),
                Message.user(
                    f"Question: {episode.question}\n\nTrajectory:\n{render_trajectory(attempt)}\n\n"
                    f"Final answer: {attempt.answer}"
                ),
            ],
        )
        reason = re.search(r"reason\s*[:：]\s*(.+)", completion.text, re.IGNORECASE)
        return read_verdict(completion.text), (reason.group(1).strip() if reason else completion.text.strip()[:300])

    def solve(self, episode: Episode) -> None:
        lessons: list[str] = []
        for trial in range(1, self.trials + 1):
            prompt = episode.question
            if lessons:
                prompt += "\n\nLessons from earlier failed attempts:\n" + "\n".join(f"- {lesson}" for lesson in lessons)
            attempt = self.actor.run(prompt)
            episode.absorb(attempt)
            episode.answer, episode.messages = attempt.answer, attempt.messages
            correct, reason = self.evaluate(episode, attempt)
            episode.events.append(
                {"trial": trial, "answer": attempt.answer, "stopped": attempt.stopped, "correct": correct, "reason": reason}
            )
            if correct or trial == self.trials:
                break
            reflection = self.call(
                episode,
                [
                    Message.system(REFLECTION_PROMPT),
                    Message.user(
                        f"Question: {episode.question}\n\nTrajectory:\n{render_trajectory(attempt)}\n\n"
                        f"Evaluator: {reason}"
                    ),
                ],
            )
            lessons.append(" ".join(reflection.text.split()))
        if episode.answer is None:
            episode.stopped = attempt.stopped
