from __future__ import annotations

import pytest

from laboratory.agents.catalog import FreeModel
from laboratory.agents.messages import Completion, Message
from laboratory.agents.providers import resolve_endpoint
from laboratory.agents.routing import ModelRouter, RouterExhausted, connect
from laboratory.agents.tools import calculator
from laboratory.agents.wire import ProviderError

ENVIRONMENT = {"GROQ_API_KEY": "k", "UNOROUTER_API_KEY": "k"}
QUESTION = [Message.user("2+2?")]


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class FakeClient:
    """Answers per endpoint label from a queue of outcomes (an exception or a text reply)."""

    def __init__(self, outcomes: dict[str, list]) -> None:
        self.outcomes = outcomes
        self.calls: list[str] = []

    def complete(self, endpoint, messages, **_):
        self.calls.append(endpoint.label)
        queue = self.outcomes.get(endpoint.label, [])
        outcome = queue.pop(0) if queue else "ok"
        if isinstance(outcome, ProviderError):
            raise outcome
        if isinstance(outcome, Completion):
            return outcome
        return Completion(Message.assistant(outcome), provider=endpoint.provider.identifier, model=endpoint.model)


def endpoint(label, rpm=None, tools=True):
    provider, model = label.split(":", 1)
    return resolve_endpoint(provider, model, ENVIRONMENT, rpm=rpm, tools=tools)


def router(endpoints, outcomes, **options):
    clock = Clock()
    client = FakeClient(outcomes)
    return ModelRouter(endpoints, client=client, clock=clock, sleep=clock.sleep, **options), client, clock


def test_fails_over_and_keeps_the_trail():
    model, _, _ = router(
        [endpoint("groq:a"), endpoint("groq:b")],
        {"groq:a": [ProviderError("rate_limited", "slow", retry_after=30.0)]},
    )
    completion = model.complete(QUESTION)
    assert completion.endpoint == "groq:b"
    assert completion.trail == ("groq:a: rate_limited: slow",)


def test_retry_after_is_honoured_then_the_endpoint_returns():
    model, _, clock = router(
        [endpoint("groq:a"), endpoint("groq:b")],
        {"groq:a": [ProviderError("rate_limited", "slow", retry_after=30.0)]},
    )
    model.complete(QUESTION)
    assert model.complete(QUESTION).endpoint == "groq:b"
    clock.now += 31
    assert model.complete(QUESTION).endpoint == "groq:a"


def test_rate_limit_without_retry_after_backs_off_exponentially():
    limited = ProviderError("rate_limited", "slow")
    model, _, clock = router([endpoint("groq:a"), endpoint("groq:b")], {"groq:a": [limited, limited]})
    model.complete(QUESTION)
    assert model.status()[0]["cooling_for"] == 30.0
    clock.now += 30
    model.complete(QUESTION)
    assert model.status()[0]["cooling_for"] == 60.0


def test_shared_limits_cool_down_the_whole_account():
    model, client, _ = router(
        [endpoint("unorouter:a"), endpoint("unorouter:b"), endpoint("groq:c")],
        {"unorouter:a": [ProviderError("rate_limited", "account-wide", retry_after=120.0)]},
    )
    assert model.complete(QUESTION).endpoint == "groq:c"
    assert client.calls == ["unorouter:a", "groq:c"]


def test_unauthorized_drops_the_provider_and_unavailable_drops_the_endpoint():
    model, _, _ = router(
        [endpoint("unorouter:a"), endpoint("unorouter:b"), endpoint("groq:c"), endpoint("groq:d")],
        {"unorouter:a": [ProviderError("unauthorized", "bad key")], "groq:c": [ProviderError("unavailable", "gone")]},
    )
    assert model.complete(QUESTION).endpoint == "groq:d"
    disabled = {row["endpoint"] for row in model.status() if row["disabled"]}
    assert disabled == {"unorouter:a", "unorouter:b", "groq:c"}


def test_bad_request_skips_the_endpoint_for_that_request_only():
    model, _, _ = router(
        [endpoint("groq:a"), endpoint("groq:b")],
        {"groq:a": [ProviderError("bad_request", "context too long")]},
    )
    assert model.complete(QUESTION).endpoint == "groq:b"
    assert model.complete(QUESTION).endpoint == "groq:a"


def test_requests_with_tools_only_reach_tool_capable_endpoints():
    model, _, _ = router([endpoint("groq:text", tools=False), endpoint("groq:tools")], {})
    assert model.complete(QUESTION, tools=[calculator()]).endpoint == "groq:tools"
    assert model.complete(QUESTION).endpoint == "groq:text"


def test_requests_are_paced_to_the_catalog_rpm():
    model, client, clock = router([endpoint("groq:a", rpm=1)], {})
    model.complete(QUESTION)
    model.complete(QUESTION)
    assert clock.slept == [pytest.approx(60.0)]
    assert client.calls == ["groq:a", "groq:a"]


def test_a_reported_empty_request_window_cools_the_endpoint():
    exhausted = Completion(Message.assistant("ok"), provider="groq", model="a", exhausted_for=45.0)
    model, _, _ = router([endpoint("groq:a"), endpoint("groq:b")], {"groq:a": [exhausted]})
    model.complete(QUESTION)
    assert model.complete(QUESTION).endpoint == "groq:b"


def test_gives_up_when_every_endpoint_cools_longer_than_max_wait():
    model, _, _ = router(
        [endpoint("groq:a")], {"groq:a": [ProviderError("rate_limited", "daily cap", retry_after=3600.0)]}, max_wait=60.0
    )
    with pytest.raises(RouterExhausted) as raised:
        model.complete(QUESTION)
    assert raised.value.trail == ("groq:a: rate_limited: daily cap",)


def test_gives_up_after_max_attempts():
    flaky = [ProviderError("transient", "502")] * 5
    model, client, _ = router([endpoint("groq:a"), endpoint("groq:b")], {"groq:a": list(flaky), "groq:b": list(flaky)}, max_attempts=3)
    with pytest.raises(RouterExhausted, match="3 attempts"):
        model.complete(QUESTION)
    assert len(client.calls) == 3


def test_connect_explains_how_to_configure_when_nothing_matches():
    catalog = [FreeModel("groq", "llama", "llama", "llama", tools=True)]
    assert [e.label for e in connect("auto", catalog=catalog, environment={"GROQ_API_KEY": "k"}, discover=False).endpoints] == ["groq:llama"]
    with pytest.raises(RouterExhausted, match="Keyless providers"):
        connect("auto", catalog=catalog, environment={}, discover=False)
