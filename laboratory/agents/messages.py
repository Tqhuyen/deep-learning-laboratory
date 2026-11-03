"""The vocabulary every agent component shares: messages, tool calls, completions.

Roles follow the OpenAI chat format: ``system``, ``user``, ``assistant`` and
``tool``. Content is always a string, so every provider accepts it as-is.
Anything that answers :meth:`ChatModel.complete` can drive an agent: the
remote :class:`~laboratory.agents.routing.ModelRouter` and the offline
:class:`~laboratory.agents.scripted.ScriptedModel` are interchangeable.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .tools import Tool


@dataclass(frozen=True)
class ToolCall:
    """A function call requested by the model; ``arguments`` is the raw JSON text."""

    identifier: str
    name: str
    arguments: str = "{}"

    def parsed_arguments(self) -> dict[str, Any]:
        """Arguments as a dict; raises ``ValueError`` if they are not a JSON object."""
        try:
            value = json.loads(self.arguments or "{}")
        except json.JSONDecodeError as error:
            raise ValueError(f"arguments are not valid JSON: {error.msg}") from None
        if not isinstance(value, dict):
            raise ValueError("arguments must be a JSON object")
        return value


@dataclass(frozen=True)
class Message:
    role: str
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    @classmethod
    def system(cls, content: str) -> Message:
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: Sequence[ToolCall] = ()) -> Message:
        return cls("assistant", content, tuple(tool_calls))

    @classmethod
    def tool(cls, call: ToolCall, content: str) -> Message:
        return cls("tool", content, tool_call_id=call.identifier, name=call.name)

    def to_record(self) -> dict[str, Any]:
        """JSON-friendly form for transcripts."""
        record: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            record["tool_calls"] = [
                {"id": call.identifier, "name": call.name, "arguments": call.arguments}
                for call in self.tool_calls
            ]
        if self.tool_call_id is not None:
            record["tool_call_id"] = self.tool_call_id
            record["name"] = self.name
        return record


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class Completion:
    """One assistant turn plus where it came from and what it cost."""

    message: Message
    reasoning: str = ""
    """Hidden reasoning the provider returned separately (or in ``<think>`` tags)."""
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    provider: str = ""
    model: str = ""
    served_model: str | None = None
    """Model id the response reported; differs from ``model`` on ``auto`` routes."""
    latency: float = 0.0
    trail: tuple[str, ...] = ()
    """Failed attempts that preceded this one, as ``provider:model: kind: message``."""
    exhausted_for: float | None = None
    """Seconds until the endpoint's request window resets, when it reported none left."""

    @property
    def text(self) -> str:
        return self.message.content

    @property
    def endpoint(self) -> str:
        return f"{self.provider}:{self.model}"

    def to_record(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "served_model": self.served_model,
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "latency": round(self.latency, 3),
            "trail": list(self.trail),
        }


class ChatModel(Protocol):
    """Anything that turns a conversation (and optional tools) into one completion."""

    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Tool] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] = (),
    ) -> Completion: ...
