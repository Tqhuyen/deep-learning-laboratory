"""Ensembles: several independent attempts, one answer.

* :class:`SelfConsistencyAgent` — Wang et al., *Self-Consistency Improves Chain
  of Thought Reasoning in Language Models* (https://arxiv.org/abs/2203.11171).
  Sample the same agent ``samples`` times and take the majority answer.
* :class:`DebateAgent` — Du et al., *Improving Factuality and Reasoning in
  Language Models through Multiagent Debate* (https://arxiv.org/abs/2305.14325).
  Several agents answer, then each revises after reading the others' answers
  for a few rounds; the last round is decided by majority.

Both wrap any agent (tools included). Diversity comes from sampling, so give
the wrapped agent a temperature above zero.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from .loops import Agent, Episode, canonical_answer, extract_final_answer


def majority_vote(answers: Sequence[str | None]) -> tuple[str | None, dict[str, int]]:
    """Most frequent answer by canonical form (ties go to the earliest); ``None`` never wins."""
    canonical = [canonical_answer(answer) for answer in answers]
    votes = Counter(form for form in canonical if form)
    if not votes:
        return None, {}
    best = max(votes.values())
    winner = next(form for form in canonical if form and votes[form] == best)
    original = next(answer for answer, form in zip(answers, canonical) if form == winner)
    return original, dict(votes)


def rationale(episode: Episode, limit: int = 300) -> str:
    """The last assistant turn without its final-answer line, shortened, for other debaters."""
    for message in reversed(episode.messages):
        if message.role == "assistant" and message.content.strip():
            text = message.content
            answer = extract_final_answer(text)
            if answer is not None:
                text = text[: text.lower().rfind("final answer")]
            text = " ".join(text.split())
            return text[-limit:]
    return ""


class SelfConsistencyAgent(Agent):
    name = "self_consistency"

    def __init__(self, agent: Agent, *, samples: int = 3):
        super().__init__(agent.model)
        if samples < 1:
            raise ValueError("self-consistency needs at least one sample")
        self.agent = agent
        self.samples = samples
        self.requires_tools = agent.requires_tools

    def solve(self, episode: Episode) -> None:
        attempts = [self.agent.run(episode.question) for _ in range(self.samples)]
        for attempt in attempts:
            episode.absorb(attempt)
        episode.answer, votes = majority_vote([attempt.answer for attempt in attempts])
        episode.events.append({"samples": [attempt.answer for attempt in attempts], "votes": votes})
        chosen = next((attempt for attempt in attempts if attempt.answer == episode.answer), attempts[0])
        episode.messages = chosen.messages
        if episode.answer is None:
            episode.stopped = chosen.stopped


class DebateAgent(Agent):
    name = "debate"

    def __init__(self, agent: Agent, *, debaters: int = 3, rounds: int = 2):
        super().__init__(agent.model)
        if debaters < 2 or rounds < 1:
            raise ValueError("a debate needs at least two debaters and one round")
        self.agent = agent
        self.debaters = debaters
        self.rounds = rounds
        self.requires_tools = agent.requires_tools

    def solve(self, episode: Episode) -> None:
        attempts = [self.agent.run(episode.question) for _ in range(self.debaters)]
        for round_number in range(1, self.rounds + 1):
            if round_number > 1:
                previous = attempts
                attempts = []
                for index in range(self.debaters):
                    others = "\n".join(
                        f"- Agent {other + 1}: {previous[other].answer} ({rationale(previous[other])})"
                        for other in range(self.debaters)
                        if other != index
                    )
                    attempts.append(
                        self.agent.run(
                            f"{episode.question}\n\nOther agents answered this question:\n{others}\n\n"
                            "Treat their answers as advice, not truth: re-check the problem (use the tools "
                            "if needed) and give your own final answer."
                        )
                    )
            for attempt in attempts:
                episode.absorb(attempt)
            episode.events.append({"round": round_number, "answers": [attempt.answer for attempt in attempts]})
        episode.answer, votes = majority_vote([attempt.answer for attempt in attempts])
        episode.events.append({"votes": votes})
        chosen = next((attempt for attempt in attempts if attempt.answer == episode.answer), attempts[0])
        episode.messages = chosen.messages
        if episode.answer is None:
            episode.stopped = chosen.stopped
