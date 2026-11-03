"""One HTTP call to one endpoint: no retries, no failover (that is the router's job).

The transport is a plain function so tests replace the network with a fake::

    client = ChatClient(transport=lambda method, url, headers, body, timeout: HttpResponse(200, {}, b"..."))

The default transport is :mod:`urllib` from the standard library, so the agent
stack adds no dependency to the laboratory.
"""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .messages import Completion, Message
from .wire import (
    ProviderError,
    chat_request,
    classify,
    exhausted_for,
    parse_chat_response,
    parse_responses_payload,
    responses_request,
    same_model,
)

if TYPE_CHECKING:
    from .providers import Endpoint
    from .tools import Tool


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    """Header names are lower-case."""
    body: bytes


Transport = Callable[[str, str, Mapping[str, str], "bytes | None", float], HttpResponse]


def urllib_transport(
    method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float
) -> HttpResponse:
    """Send one request; HTTP errors are returned, network failures raise ``transient``."""
    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            lowered = {name.lower(): value for name, value in response.headers.items()}
            return HttpResponse(response.status, lowered, response.read())
    except urllib.error.HTTPError as error:
        lowered = {name.lower(): value for name, value in (error.headers or {}).items()}
        return HttpResponse(error.code, lowered, error.read())
    except (urllib.error.URLError, http.client.HTTPException, OSError) as error:
        reason = getattr(error, "reason", None) or error
        raise ProviderError("transient", f"network error: {reason}") from None


class ChatClient:
    """Encode, send and decode one completion request for any :class:`Endpoint`."""

    def __init__(
        self,
        transport: Transport = urllib_transport,
        *,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.transport = transport
        self.clock = clock
        self.wall_clock = wall_clock

    def complete(
        self,
        endpoint: Endpoint,
        messages: Sequence[Message],
        *,
        tools: Sequence[Tool] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] = (),
    ) -> Completion:
        provider = endpoint.provider
        if provider.wire == "responses":
            url = f"{endpoint.base_url}/responses"
            body = responses_request(endpoint, messages, tools, temperature, max_tokens)
            parse = parse_responses_payload
        else:
            url = f"{endpoint.base_url}/chat/completions"
            body = chat_request(endpoint, messages, tools, temperature, max_tokens, stop)
            parse = parse_chat_response

        started = self.clock()
        response = self.transport("POST", url, endpoint.headers(), json.dumps(body).encode("utf-8"), provider.timeout)
        latency = self.clock() - started
        if not 200 <= response.status < 300:
            raise classify(response.status, response.headers, response.body, self.wall_clock())
        try:
            payload = json.loads(response.body.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            raise ProviderError("transient", "200 response with a non-JSON body", status=response.status) from None

        message, reasoning, finish_reason, usage, served_model = parse(payload)
        served_model = response.headers.get("x-routed-via") or served_model
        if provider.verify_model and not endpoint.model.startswith("auto") and not same_model(endpoint.model, served_model):
            raise ProviderError("unavailable", f"served {served_model!r} instead of {endpoint.model!r}")
        if not message.content.strip() and not message.tool_calls:
            hint = "; the hidden reasoning used up max_tokens" if finish_reason == "length" else ""
            raise ProviderError("bad_request", f"empty completion (finish_reason={finish_reason}){hint}")
        return Completion(
            message=message,
            reasoning=reasoning,
            finish_reason=finish_reason,
            usage=usage,
            provider=provider.identifier,
            model=endpoint.model,
            served_model=served_model,
            latency=latency,
            exhausted_for=exhausted_for(response.headers, self.wall_clock()),
        )

    def list_models(self, endpoint: Endpoint) -> list[str]:
        """Model ids from ``GET {base}/models`` (the endpoint's model is ignored)."""
        response = self.transport("GET", f"{endpoint.base_url}/models", endpoint.headers(), None, endpoint.provider.timeout)
        if not 200 <= response.status < 300:
            raise classify(response.status, response.headers, response.body, self.wall_clock())
        try:
            payload = json.loads(response.body.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            raise ProviderError("transient", "model list is not JSON") from None
        rows = payload.get("data", []) if isinstance(payload, dict) else payload
        return [str(row["id"]) for row in rows if isinstance(row, dict) and "id" in row]
