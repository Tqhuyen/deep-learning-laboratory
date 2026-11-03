"""Single-agent loops: the baselines every other architecture is built from.

* :class:`DirectAgent` — one call, no tools. The floor any architecture must beat.
* :class:`ReActAgent` — Yao et al., *ReAct: Synergizing Reasoning and Acting
  in Language Models* (https://arxiv.org/abs/2210.03629). Thought / Action /
  Observation as plain text, so it runs on **any** chat model, including free
  models without function calling.
* :class:`ToolCallingAgent` — the same loop over native function calling
  (OpenAI ``tools``): the model emits structured calls, tool results return as
  ``tool`` messages. Needs a model with tool support.

Every architecture returns an :class:`Episode`: the answer, why it stopped, the
transcript, every completion (endpoint, tokens, latency) and an ``events`` list
for architecture-specific traces (plans, verdicts, votes). Model failures end
the episode with ``stopped="error"`` instead of raising, so one exhausted
endpoint does not abort a whole comparison.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .messages import ChatModel, Completion, Message, Usage

if TYPE_CHECKING:
    from .tools import Tool, Toolbox


# --- answers ----------------------------------------------------------------------

_FINAL_ANSWER = re.compile(r"final\s+answer\**\s*[:：]\s*\**", re.IGNORECASE)
_NUMBER = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")


def extract_final_answer(text: str) -> str | None:
    """The first line after the last ``Final Answer:`` marker (markdown and ``\\boxed`` removed)."""
    markers = list(_FINAL_ANSWER.finditer(text))
    if not markers:
        return None
    lines = [line.strip() for line in text[markers[-1].end() :].splitlines() if line.strip()]
    if not lines:
        return None
    answer = lines[0]
    boxed = re.search(r"\\boxed\{([^{}]*)\}", answer)
    if boxed:
        answer = boxed.group(1)
    return answer.strip("*`$ ").strip() or None


def last_line(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else None


def numbers_in(text: str) -> list[float]:
    """Every number in ``text``; ``$1,234.50`` reads as 1234.5."""
    values = []
    for match in _NUMBER.finditer(text):
        try:
            values.append(float(match.group().replace("$", "").replace(",", "")))
        except ValueError:
            continue
    return values


def canonical_answer(text: str | None) -> str | None:
    """Comparable form of an answer: a number when it is one, else lower-case words without articles."""
    if text is None:
        return None
    stripped = text.strip().strip("*`").rstrip(".")
    numbers = numbers_in(stripped)
    words = re.findall(r"[a-z]+", stripped.lower())
    if len(numbers) == 1 and len(words) <= 3:
        return format(round(numbers[0], 6), "f").rstrip("0").rstrip(".")
    tokens = re.findall(r"[a-z0-9]+", stripped.lower())
    content = [token for token in tokens if token not in {"a", "an", "the"}]
    return " ".join(content or tokens)


# --- episodes ---------------------------------------------------------------------


@dataclass
class Episode:
    architecture: str
    question: str
    answer: str | None = None
    stopped: str = "answer"
    """``"answer"``, ``"max_steps"`` or ``"error"``."""
    error: str | None = None
    messages: list[Message] = field(default_factory=list)
    completions: list[Completion] = field(default_factory=list)
    tool_calls: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def calls(self) -> int:
        return len(self.completions)

    @property
    def usage(self) -> Usage:
        total = Usage()
        for completion in self.completions:
            total = total + completion.usage
        return total

    @property
    def latency(self) -> float:
        return sum(completion.latency for completion in self.completions)

    def absorb(self, other: Episode) -> None:
        """Count a sub-episode's calls and tool calls as this episode's own."""
        self.completions.extend(other.completions)
        self.tool_calls += other.tool_calls

    def to_record(self) -> dict[str, Any]:
        usage = self.usage
        return {
            "architecture": self.architecture,
            "question": self.question,
            "answer": self.answer,
            "stopped": self.stopped,
            "error": self.error,
            "calls": self.calls,
            "tool_calls": self.tool_calls,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "latency": round(self.latency, 3),
            "completions": [completion.to_record() for completion in self.completions],
            "messages": [message.to_record() for message in self.messages],
            "events": self.events,
        }


class Agent:
    """Base class: ``run`` wraps ``solve`` so model failures end the episode cleanly."""

    name = "agent"
    requires_tools = False
    """True when the model must support native function calling."""

    def __init__(self, model: ChatModel, *, temperature: float | None = None, max_tokens: int | None = None):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run(self, question: str) -> Episode:
        episode = Episode(self.name, question)
        try:
            self.solve(episode)
        except RuntimeError as error:  # ProviderError, RouterExhausted, an exhausted script
            episode.stopped = "error"
            episode.error = f"{type(error).__name__}: {error}"
        return episode

    def solve(self, episode: Episode) -> None:
        raise NotImplementedError

    def call(
        self, episode: Episode, messages: Sequence[Message], *, tools: Sequence[Tool] = (), stop: Sequence[str] = ()
    ) -> Completion:
        completion = self.model.complete(
            messages, tools=tools, temperature=self.temperature, max_tokens=self.max_tokens, stop=stop
        )
        episode.completions.append(completion)
        return completion


# --- direct -----------------------------------------------------------------------

DIRECT_PROMPT = (
    "Answer the question. Reason briefly if it helps, then finish with a final line of the form\n"
    "Final Answer: <the answer only>"
)


class DirectAgent(Agent):
    name = "direct"

    def solve(self, episode: Episode) -> None:
        episode.messages = [Message.system(DIRECT_PROMPT), Message.user(episode.question)]
        completion = self.call(episode, episode.messages)
        episode.messages.append(completion.message)
        episode.answer = extract_final_answer(completion.text) or last_line(completion.text)


# --- ReAct ------------------------------------------------------------------------

REACT_PROMPT = """Answer the question by interleaving Thought, Action and Observation steps.

You can use these tools:
{tools}

Use exactly this format:

Thought: what you know so far and what to do next
Action: the tool to use, one of [{names}]
Action Input: the tool arguments as a JSON object
Observation: the tool result (it is written for you; never write it yourself)
... (Thought / Action / Action Input / Observation may repeat)
Thought: I now know the answer
Final Answer: the answer only, without explanation"""

REACT_FORMAT_REMINDER = (
    "Observation: I could not find an Action. Reply with 'Action:' and 'Action Input:' lines, "
    "or with 'Final Answer:' when you know the answer."
)

_ACTION = re.compile(r"^\s*\**Action\s*\d*\**\s*:\s*\**(?P<action>[^\n]*)", re.IGNORECASE | re.MULTILINE)
_ACTION_INPUT = re.compile(r"Action\s*\d*\s*Input\s*\d*\**\s*:\s*\**(?P<input>.*)", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class Action:
    tool: str
    arguments: str
    position: int


def _first_json_object(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", text.strip())
    if text.startswith("{"):
        try:
            value, _ = json.JSONDecoder().raw_decode(text)
            return json.dumps(value)
        except json.JSONDecodeError:
            pass
    first = text.splitlines()[0].strip() if text else ""
    return first.strip("`")


def parse_action(text: str) -> Action | None:
    """Read ``Action: tool`` + ``Action Input: {...}``, or the paper's ``Action: tool[input]``."""
    match = _ACTION.search(text)
    if match is None:
        return None
    action = match.group("action").strip().strip("*`").strip()
    bracketed = re.fullmatch(r"([\w.\-]+)\s*[\[(](.*)[\])]", action, re.DOTALL)
    if bracketed:
        return Action(bracketed.group(1), _first_json_object(bracketed.group(2)), match.start())
    tool = action.split()[0].strip("`*'\"") if action else ""
    argument_match = _ACTION_INPUT.search(text, match.end())
    arguments = _first_json_object(argument_match.group("input")) if argument_match else ""
    return Action(tool, arguments, match.start()) if tool else None


def truncate_at_observation(text: str) -> str:
    """Drop anything from a self-written ``Observation:`` on (models hallucinate tool results)."""
    match = re.search(r"^\s*\**Observation\**\s*:", text, re.IGNORECASE | re.MULTILINE)
    return text[: match.start()].rstrip() if match else text.rstrip()


class ReActAgent(Agent):
    name = "react"

    def __init__(self, model: ChatModel, toolbox: Toolbox, *, max_steps: int = 8, **sampling):
        super().__init__(model, **sampling)
        self.toolbox = toolbox
        self.max_steps = max_steps

    def solve(self, episode: Episode) -> None:
        system = REACT_PROMPT.format(tools=self.toolbox.describe(), names=", ".join(self.toolbox.names))
        episode.messages = [Message.system(system), Message.user(f"Question: {episode.question}")]
        for _ in range(self.max_steps):
            completion = self.call(episode, episode.messages, stop=("\nObservation:",))
            text = truncate_at_observation(completion.text)
            episode.messages.append(Message.assistant(text))
            action = parse_action(text)
            final = _FINAL_ANSWER.search(text)
            answer = extract_final_answer(text) if final and (action is None or final.start() < action.position) else None
            if answer is not None:
                episode.answer = answer
                return
            if action is None:
                episode.messages.append(Message.user(REACT_FORMAT_REMINDER))
                continue
            observation = self.toolbox.run(action.tool, action.arguments)
            episode.tool_calls += 1
            episode.messages.append(Message.user(f"Observation: {observation}"))
        episode.stopped = "max_steps"


# --- native function calling ------------------------------------------------------

TOOL_CALLING_PROMPT = (
    "Answer the question. Call the tools whenever they help; you may call them several times. "
    "When you know the answer, reply with a final line of the form\nFinal Answer: <the answer only>"
)


class ToolCallingAgent(Agent):
    name = "tool_calling"
    requires_tools = True

    def __init__(self, model: ChatModel, toolbox: Toolbox, *, max_steps: int = 8, **sampling):
        super().__init__(model, **sampling)
        self.toolbox = toolbox
        self.max_steps = max_steps

    def solve(self, episode: Episode) -> None:
        episode.messages = [Message.system(TOOL_CALLING_PROMPT), Message.user(episode.question)]
        for _ in range(self.max_steps):
            completion = self.call(episode, episode.messages, tools=self.toolbox.tools)
            episode.messages.append(completion.message)
            if not completion.message.tool_calls:
                episode.answer = extract_final_answer(completion.text) or last_line(completion.text)
                return
            for call in completion.message.tool_calls:
                episode.messages.append(Message.tool(call, self.toolbox.run(call.name, call.arguments)))
                episode.tool_calls += 1
        episode.stopped = "max_steps"
