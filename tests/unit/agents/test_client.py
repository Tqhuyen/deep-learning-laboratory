from __future__ import annotations

import json
from itertools import count

import pytest

from laboratory.agents.client import ChatClient, HttpResponse
from laboratory.agents.messages import Message
from laboratory.agents.providers import resolve_endpoint
from laboratory.agents.wire import ProviderError


def recording_transport(*responses: HttpResponse):
    queue = list(responses)
    requests: list[dict] = []

    def transport(method, url, headers, body, timeout):
        requests.append({"method": method, "url": url, "headers": dict(headers), "body": json.loads(body) if body else None})
        return queue.pop(0)

    return transport, requests


def ok(payload: dict, **headers) -> HttpResponse:
    return HttpResponse(200, {name.replace("_", "-"): value for name, value in headers.items()}, json.dumps(payload).encode())


def reply(content: str, model: str = "m") -> dict:
    return {"model": model, "choices": [{"message": {"content": content}, "finish_reason": "stop"}]}


GROQ = resolve_endpoint("groq", "llama", {"GROQ_API_KEY": "secret"})


def test_posts_chat_completions_and_measures_latency_with_the_injected_clock():
    transport, requests = recording_transport(ok(reply("hi")))
    ticks = count(start=10.0, step=1.5)
    completion = ChatClient(transport, clock=lambda: next(ticks)).complete(GROQ, [Message.user("hello")], max_tokens=32)
    assert requests[0]["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert requests[0]["headers"]["Authorization"] == "Bearer secret"
    assert requests[0]["body"]["max_tokens"] == 32
    assert (completion.text, completion.endpoint, completion.latency) == ("hi", "groq:llama", 1.5)


def test_responses_wire_goes_to_the_responses_route():
    aclide = resolve_endpoint("aclide", "qwen", {"ACLIDE_API_KEY": "k"})
    payload = {"model": "qwen", "output": [{"type": "message", "content": [{"type": "output_text", "text": "pong"}]}]}
    transport, requests = recording_transport(ok(payload))
    assert ChatClient(transport).complete(aclide, [Message.user("ping")]).text == "pong"
    assert requests[0]["url"] == "https://aclide.com/v1/responses"


def test_http_errors_are_classified():
    transport, _ = recording_transport(HttpResponse(429, {"retry-after": "7"}, b'{"error": {"message": "slow"}}'))
    with pytest.raises(ProviderError) as raised:
        ChatClient(transport).complete(GROQ, [Message.user("x")])
    assert (raised.value.kind, raised.value.retry_after) == ("rate_limited", 7.0)


def test_gateways_that_substitute_models_are_caught():
    septor = resolve_endpoint("septor", "llama-3.3-70b", {"SEPTOR_API_KEY": "k"})
    transport, _ = recording_transport(ok(reply("hi", model="minimax-m2.5-free")), ok(reply("hi", model="llama-3.3-70b")))
    client = ChatClient(transport)
    with pytest.raises(ProviderError) as raised:
        client.complete(septor, [Message.user("x")])
    assert raised.value.kind == "unavailable"
    assert client.complete(septor, [Message.user("x")]).served_model == "llama-3.3-70b"


def test_routed_via_header_names_the_model_behind_an_auto_route():
    router = resolve_endpoint("freellmapi", "auto", {"FREELLMAPI_API_KEY": "k"})
    transport, _ = recording_transport(ok(reply("hi", model="auto"), x_routed_via="groq/llama-3.3-70b"))
    assert ChatClient(transport).complete(router, [Message.user("x")]).served_model == "groq/llama-3.3-70b"


@pytest.mark.parametrize(
    ("response", "kind"),
    [
        (ok({"model": "m", "choices": [{"message": {"content": "   "}, "finish_reason": "length"}]}), "bad_request"),
        (HttpResponse(200, {}, b"<html>not json</html>"), "transient"),
    ],
)
def test_unusable_successful_responses_raise(response, kind):
    transport, _ = recording_transport(response)
    with pytest.raises(ProviderError) as raised:
        ChatClient(transport).complete(GROQ, [Message.user("x")])
    assert raised.value.kind == kind


def test_exhausted_request_window_is_reported():
    transport, _ = recording_transport(ok(reply("hi"), x_ratelimit_remaining_requests="0", x_ratelimit_reset_requests="45s"))
    assert ChatClient(transport).complete(GROQ, [Message.user("x")]).exhausted_for == 45.0


def test_model_listing_reads_ids():
    transport, requests = recording_transport(ok({"data": [{"id": "a"}, {"id": "b"}, {"object": "no id"}]}))
    assert ChatClient(transport).list_models(GROQ) == ["a", "b"]
    assert requests[0]["method"] == "GET" and requests[0]["url"].endswith("/models")
