"""Bytes <-> objects for the two wire protocols, and what an HTTP failure means.

* ``chat`` — OpenAI Chat Completions, ``POST {base}/chat/completions``. 36 of
  the 37 free providers speak it, each with the small divergences recorded on
  :class:`~laboratory.agents.providers.Provider`.
* ``responses`` — the OpenAI Responses API, ``POST {base}/responses`` (ACLIDE).

Every failure becomes a :class:`ProviderError` whose ``kind`` tells the router
what to do next:

=================  =========================================  ==========================
kind               typical cause                              router action
=================  =========================================  ==========================
``rate_limited``   429, UnoRouter 503 ``get_channel_failed``  cool down, honour Retry-After
``transient``      5xx, timeout, malformed or in-band error   short cool-down, fail over
``unauthorized``   401, "API key not valid"                   drop the provider
``unavailable``    402/403/404, unknown or premium model      drop the endpoint
``bad_request``    400/413/422, request this model rejects    fail over for this request
=================  =========================================  ==========================

Everything here is pure (no I/O), so every rule is unit-tested offline.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any

from .messages import Message, ToolCall, Usage

if TYPE_CHECKING:
    from .providers import Endpoint, Provider
    from .tools import Tool

ERROR_KINDS = ("rate_limited", "transient", "unauthorized", "unavailable", "bad_request")
PROXY_ERROR_PREFIX = "### **Proxy error (HTTP "  # ElectronHub reports broken upstreams inside a 200

_AUTHENTICATION = re.compile(
    r"api[ _-]?key (is )?(not valid|invalid)|invalid (api[ _-]?)?(key|token)|incorrect api key|"
    r"unauthori[sz]ed|authentication (error|failed|required)|missing (api[ _-]?)?key|no api key",
    re.IGNORECASE,
)
_PAYMENT = re.compile(
    r"insufficient (balance|credits?|funds)|payment required|requires? (a )?(paid|subscription)|"
    r"premium model|phone verification|free tier can only",
    re.IGNORECASE,
)
_MISSING_MODEL = re.compile(
    r"model_not_found|no such model|unknown model|model[^.]{0,80}(not found|does not exist|not exist|"
    r"not available|not supported|not offered|deprecated|decommissioned|retired)",
    re.IGNORECASE,
)
_DURATION = re.compile(
    r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?\Z"
)
_RESET_HEADERS = ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset", "ratelimit-reset")
_REMAINING_TO_RESET = (
    ("x-ratelimit-remaining-requests", "x-ratelimit-reset-requests"),
    ("x-ratelimit-remaining", "x-ratelimit-reset"),
    ("ratelimit-remaining", "ratelimit-reset"),
    ("x-ratelimit-remaining-minute", None),
)


class ProviderError(RuntimeError):
    """A failed call, classified so the router knows whether and when to retry."""

    def __init__(self, kind: str, message: str, *, status: int | None = None, retry_after: float | None = None):
        if kind not in ERROR_KINDS:
            raise ValueError(f"unknown error kind {kind!r}")
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.message = message
        self.status = status
        self.retry_after = retry_after


# --- requests -------------------------------------------------------------------


def strip_schema_keys(schema: Any, keys: frozenset[str]) -> Any:
    """Copy of a JSON schema without ``keys`` at any depth."""
    if isinstance(schema, dict):
        return {name: strip_schema_keys(value, keys) for name, value in schema.items() if name not in keys}
    if isinstance(schema, list):
        return [strip_schema_keys(value, keys) for value in schema]
    return schema


def _max_tokens(provider: Provider, requested: int | None) -> int | None:
    value = requested if requested is not None else provider.default_max_tokens
    if value is not None and provider.minimum_max_tokens is not None:
        value = max(value, provider.minimum_max_tokens)
    return value


def _check_tools(endpoint: Endpoint, tools: Sequence[Tool]) -> None:
    if tools and not endpoint.tools:
        raise ProviderError("bad_request", f"{endpoint.label} does not support tool calling")


def _chat_message(message: Message) -> dict[str, Any]:
    if message.role == "tool":
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content}
    wire: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        wire["tool_calls"] = [
            {"id": call.identifier, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
            for call in message.tool_calls
        ]
    return wire


def chat_request(
    endpoint: Endpoint,
    messages: Sequence[Message],
    tools: Sequence[Tool] = (),
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop: Sequence[str] = (),
) -> dict[str, Any]:
    """JSON body for ``POST {base}/chat/completions``."""
    provider = endpoint.provider
    _check_tools(endpoint, tools)
    body: dict[str, Any] = {"model": endpoint.model, "messages": [_chat_message(m) for m in messages]}
    if tools:
        body["tools"] = [strip_schema_keys(tool.schema(), provider.strip_schema_keys) for tool in tools]
        if provider.single_tool_call:
            body["parallel_tool_calls"] = False
    if temperature is not None:
        body["temperature"] = temperature
    limit = _max_tokens(provider, max_tokens)
    if limit is not None:
        body[provider.max_tokens_field] = limit
    if stop and provider.stop:
        body["stop"] = list(stop)[:4]
    return body


def responses_request(
    endpoint: Endpoint,
    messages: Sequence[Message],
    tools: Sequence[Tool] = (),
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """JSON body for ``POST {base}/responses``; tool turns become typed input items."""
    provider = endpoint.provider
    _check_tools(endpoint, tools)
    items: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "tool":
            items.append({"type": "function_call_output", "call_id": message.tool_call_id, "output": message.content})
            continue
        if message.content or not message.tool_calls:
            items.append({"role": message.role, "content": message.content})
        for call in message.tool_calls:
            items.append({"type": "function_call", "call_id": call.identifier, "name": call.name, "arguments": call.arguments})
    body: dict[str, Any] = {"model": endpoint.model, "input": items, "store": False}
    if tools:
        functions = (strip_schema_keys(tool.schema(), provider.strip_schema_keys)["function"] for tool in tools)
        body["tools"] = [{"type": "function", **function} for function in functions]
    if temperature is not None:
        body["temperature"] = temperature
    limit = _max_tokens(provider, max_tokens)
    if limit is not None:
        body["max_output_tokens"] = limit
    return body


# --- responses ------------------------------------------------------------------


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_text(part.get("text")) if isinstance(part, dict) else str(part) for part in value)
    if isinstance(value, dict):
        return _text(value.get("text") or value.get("content"))
    return str(value)


def split_thinking(content: str) -> tuple[str, str]:
    """Separate a leading ``<think>...</think>`` block: returns ``(reasoning, answer)``."""
    match = re.match(r"\s*<think>(.*?)</think>", content, re.DOTALL)
    if match:
        return match.group(1).strip(), content[match.end() :].lstrip()
    if "</think>" in content and "<think>" not in content:
        reasoning, _, answer = content.rpartition("</think>")
        return reasoning.strip(), answer.lstrip()
    return "", content


def _usage(raw: Any) -> Usage:
    if not isinstance(raw, dict):
        return Usage()
    prompt = raw.get("prompt_tokens", raw.get("input_tokens", 0))
    completion = raw.get("completion_tokens", raw.get("output_tokens", 0))
    return Usage(int(prompt or 0), int(completion or 0))


def _raise_embedded_error(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ProviderError("transient", "response body is not a JSON object")
    if payload.get("error"):
        message, _ = error_details(json.dumps(payload).encode())
        raise ProviderError("transient", f"error inside a successful response: {message}")


def parse_chat_response(payload: Any) -> tuple[Message, str, str | None, Usage, str | None]:
    """``(message, reasoning, finish_reason, usage, served_model)`` from a chat response."""
    _raise_embedded_error(payload)
    choices = payload.get("choices") or []
    if not choices:
        raise ProviderError("transient", "response has no choices")
    choice = choices[0]
    raw = choice.get("message") or {}
    reasoning, content = split_thinking(_text(raw.get("content")))
    reasoning = _text(raw.get("reasoning_content")) or _text(raw.get("reasoning")) or reasoning
    if content.lstrip().startswith(PROXY_ERROR_PREFIX):
        raise ProviderError("transient", "upstream proxy error returned inside the completion")
    calls = []
    for index, call in enumerate(raw.get("tool_calls") or []):
        function = call.get("function") or {}
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments if arguments is not None else {})
        calls.append(ToolCall(call.get("id") or f"call_{index}", function.get("name", ""), arguments or "{}"))
    message = Message.assistant(content, calls)
    return message, reasoning, choice.get("finish_reason"), _usage(payload.get("usage")), payload.get("model")


def parse_responses_payload(payload: Any) -> tuple[Message, str, str | None, Usage, str | None]:
    """Same tuple as :func:`parse_chat_response`, from a Responses API object."""
    _raise_embedded_error(payload)
    status = payload.get("status")
    if status == "failed":
        raise ProviderError("transient", "response status is 'failed'")
    texts: list[str] = []
    reasoning: list[str] = []
    calls: list[ToolCall] = []
    for item in payload.get("output") or []:
        kind = item.get("type")
        if kind == "message":
            texts.extend(_text(part.get("text")) for part in item.get("content") or [] if isinstance(part, dict))
        elif kind == "function_call":
            identifier = item.get("call_id") or item.get("id") or f"call_{len(calls)}"
            calls.append(ToolCall(identifier, item.get("name", ""), item.get("arguments") or "{}"))
        elif kind == "reasoning":
            reasoning.extend(_text(part.get("text")) for part in item.get("summary") or [] if isinstance(part, dict))
    finish = "tool_calls" if calls else ("length" if status == "incomplete" else "stop")
    message = Message.assistant("".join(texts), calls)
    return message, "\n".join(reasoning), finish, _usage(payload.get("usage")), payload.get("model")


def same_model(requested: str, served: str | None) -> bool:
    """Whether a gateway served the requested weights (vendor prefix and ``:free`` ignored)."""

    def normalize(model: str) -> str:
        name = model.rsplit("/", 1)[-1].lower().removesuffix(":free")
        return re.sub(r"[^a-z0-9]", "", name)

    return bool(served) and normalize(requested) == normalize(served)


# --- failures -------------------------------------------------------------------


def error_details(body: bytes) -> tuple[str, str | None]:
    """Human message and machine code from an error body of any common shape."""
    text = body.decode("utf-8", "replace").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return (text[:300] or "(empty body)"), None
    if not isinstance(payload, dict):
        return text[:300], None
    error = payload.get("error")
    if isinstance(error, dict):
        code = error.get("code") or error.get("type")
        return str(error.get("message") or error), (str(code) if code is not None else None)
    for key in ("error", "detail", "message", "title"):
        if payload.get(key):
            return _text(payload[key]) if not isinstance(payload[key], list) else json.dumps(payload[key]), None
    errors = payload.get("errors")
    if isinstance(errors, list) and errors and isinstance(errors[0], dict):
        return str(errors[0].get("message", errors[0])), None
    return text[:300], None


def seconds_until(value: str, now: float) -> float | None:
    """Parse a reset header: seconds, ``1m30s``-style durations, epoch s/ms or an HTTP date."""
    value = value.strip()
    if not value:
        return None
    try:
        number = float(value)
    except ValueError:
        match = _DURATION.match(value)
        if match and any(match.groups()):
            hours, minutes, seconds, milliseconds = (float(group) if group else 0.0 for group in match.groups())
            return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - now)
        except (TypeError, ValueError):
            return None
    if number > 1e12:
        return max(0.0, number / 1000 - now)
    if number > 1e9:
        return max(0.0, number - now)
    return max(0.0, number)


def retry_after(headers: Mapping[str, str], now: float) -> float | None:
    for name in _RESET_HEADERS:
        if name in headers:
            seconds = seconds_until(headers[name], now)
            if seconds is not None:
                return seconds
    return None


def exhausted_for(headers: Mapping[str, str], now: float) -> float | None:
    """Seconds to wait when a successful response says no requests remain."""
    for remaining, reset in _REMAINING_TO_RESET:
        value = headers.get(remaining, "").strip()
        if value and value.split(".")[0] == "0":
            seconds = seconds_until(headers.get(reset, ""), now) if reset else None
            return seconds if seconds is not None else 60.0
    return None


def classify(status: int, headers: Mapping[str, str], body: bytes, now: float) -> ProviderError:
    """Turn a non-2xx response into the :class:`ProviderError` the router acts on."""
    message, code = error_details(body)
    detail = f"HTTP {status}: {message}"
    wait = retry_after(headers, now)
    if status == 401 or (400 <= status < 500 and _AUTHENTICATION.search(message)):
        return ProviderError("unauthorized", detail, status=status)
    if status == 429:
        if _PAYMENT.search(message):
            return ProviderError("unavailable", detail, status=status)
        return ProviderError("rate_limited", detail, status=status, retry_after=wait)
    if code == "get_channel_failed":
        return ProviderError("rate_limited", detail, status=status, retry_after=wait)
    if status in (402, 403, 404) or code == "model_not_found" or _PAYMENT.search(message):
        return ProviderError("unavailable", detail, status=status)
    if status >= 500 or status in (408, 409, 425):
        return ProviderError("transient", detail, status=status, retry_after=wait)
    if _MISSING_MODEL.search(message):
        return ProviderError("unavailable", detail, status=status)
    return ProviderError("bad_request", detail, status=status)
