"""An offline :class:`~laboratory.agents.messages.ChatModel` that replays a script.

It is what the ``tiny`` tier and the unit tests run on: no network, no keys,
fully deterministic, yet every architecture executes its real control flow
(prompting, parsing, tool dispatch, stopping). Each call consumes the next
step of the script; a step is

* assistant text,
* a :class:`~laboratory.agents.messages.Message` (to script tool calls, see
  :func:`tool_call_message`),
* a :class:`~laboratory.agents.messages.Completion`, or
* a callable ``step(messages, tools)`` returning one of the above, for replies
  that depend on the conversation (for example, repeating the last observation).

Every request is recorded in :attr:`ScriptedModel.requests`, so a test can
assert what an architecture actually sent.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any

from .messages import Completion, Message, ToolCall

if TYPE_CHECKING:
    from .tools import Tool

Step = str | Message | Completion | Callable[[list[Message], list["Tool"]], Any]


def tool_call_message(name: str, arguments: dict[str, Any], identifier: str = "call_0", content: str = "") -> Message:
    """An assistant turn that calls ``name`` with ``arguments`` (native function calling)."""
    return Message.assistant(content, [ToolCall(identifier, name, json.dumps(arguments))])


class ScriptedModel:
    def __init__(self, script: Iterable[Step], *, name: str = "scripted") -> None:
        self.script = list(script)
        self.name = name
        self.requests: list[dict[str, Any]] = []

    @property
    def remaining(self) -> int:
        return len(self.script) - len(self.requests)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Tool] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] = (),
    ) -> Completion:
        if self.remaining <= 0:
            raise RuntimeError(f"script exhausted after {len(self.script)} completions")
        step = self.script[len(self.requests)]
        self.requests.append(
            {
                "messages": list(messages),
                "tools": [tool.name for tool in tools],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stop": list(stop),
            }
        )
        if callable(step):
            step = step(list(messages), list(tools))
        if isinstance(step, Completion):
            return step
        message = step if isinstance(step, Message) else Message.assistant(str(step))
        return Completion(
            message=message,
            finish_reason="tool_calls" if message.tool_calls else "stop",
            provider="scripted",
            model=self.name,
            served_model=self.name,
        )
