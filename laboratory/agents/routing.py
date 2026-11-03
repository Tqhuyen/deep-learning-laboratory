"""Failover across free endpoints: the piece that makes free tiers usable.

One free endpoint is a toy: 1-30 requests per minute, a daily cap, outages.
:class:`ModelRouter` holds endpoints in priority order and, per request, calls
the first one that is ready. On failure it reacts to the error kind (see
:mod:`laboratory.agents.wire`):

* ``rate_limited`` — cool the endpoint down for Retry-After, or 30 s doubling
  per consecutive strike; on shared-limit providers (UnoRouter, OpenRouter) the
  whole account cools down;
* ``transient`` — short cool-down, then the endpoint is eligible again;
* ``unauthorized`` — every endpoint of that provider is dropped;
* ``unavailable`` — that endpoint is dropped for the rest of the run;
* ``bad_request`` — skipped for this request only (it may suit other requests:
  a reasoning model that overruns ``max_tokens`` on one prompt answers others).

Each endpoint is also paced to its catalog RPM, and a response reporting zero
remaining requests cools it down until its window resets. When every candidate
is cooling down the router sleeps until the first is ready (up to
``max_wait``), so long runs ride through per-minute caps instead of failing.

Pin a model **family** when comparing agent architectures: failover then
changes the provider, never the weights.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from .catalog import FreeModel, load_catalog, select_endpoints
from .client import ChatClient
from .environment import load_environment
from .messages import Completion, Message
from .providers import CATALOG_PROVIDERS, PROVIDERS, Endpoint, resolve_endpoint
from .wire import ProviderError

if TYPE_CHECKING:
    from .tools import Tool

logger = logging.getLogger(__name__)

RATE_LIMIT_BACKOFF = 30.0
MAXIMUM_COOLDOWN = 3600.0


class RouterExhausted(RuntimeError):
    """No endpoint could serve the request; ``trail`` lists every failed attempt."""

    def __init__(self, message: str, trail: Sequence[str] = ()):
        details = "".join(f"\n  - {attempt}" for attempt in trail)
        super().__init__(message + details)
        self.trail = tuple(trail)


@dataclass
class _State:
    cooldown_until: float = 0.0
    disabled: str | None = None
    strikes: int = 0
    calls: deque[float] = field(default_factory=deque)


class ModelRouter:
    """A :class:`~laboratory.agents.messages.ChatModel` over endpoints with failover."""

    def __init__(
        self,
        endpoints: Sequence[Endpoint],
        *,
        client: ChatClient | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        max_attempts: int = 8,
        max_wait: float = 180.0,
        transient_cooldown: float = 20.0,
    ) -> None:
        if not endpoints:
            raise ValueError("ModelRouter needs at least one endpoint")
        self.endpoints = list(endpoints)
        self.client = client or ChatClient(clock=clock)
        self.clock = clock
        self.sleep = sleep
        self.max_attempts = max_attempts
        self.max_wait = max_wait
        self.transient_cooldown = transient_cooldown
        self._states = {endpoint.label: _State() for endpoint in self.endpoints}

    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Tool] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] = (),
    ) -> Completion:
        trail: list[str] = []
        skipped: set[str] = set()
        attempts = 0
        while attempts < self.max_attempts:
            candidates = [
                endpoint
                for endpoint in self.endpoints
                if not self._states[endpoint.label].disabled
                and (endpoint.tools or not tools)
                and endpoint.label not in skipped
            ]
            if not candidates:
                raise RouterExhausted("no endpoint can serve this request", trail)
            now = self.clock()
            waits = [self._ready_in(endpoint, now) for endpoint in candidates]
            ready = [endpoint for endpoint, wait in zip(candidates, waits) if wait <= 0]
            if not ready:
                wait = min(waits)
                if wait > self.max_wait:
                    raise RouterExhausted(f"every endpoint is cooling down for at least {wait:.0f}s", trail)
                logger.info("all %d endpoints are cooling down; waiting %.0fs", len(candidates), wait)
                self.sleep(wait)
                continue

            endpoint = ready[0]
            state = self._states[endpoint.label]
            state.calls.append(self.clock())
            attempts += 1
            try:
                completion = self.client.complete(
                    endpoint, messages, tools=tools, temperature=temperature, max_tokens=max_tokens, stop=stop
                )
            except ProviderError as error:
                trail.append(f"{endpoint.label}: {error.kind}: {error.message}"[:400])
                logger.info("%s -> %s: %s", endpoint.label, error.kind, error.message[:200])
                self._penalize(endpoint, error, skipped)
                continue
            state.strikes = 0
            if completion.exhausted_for:
                state.cooldown_until = max(state.cooldown_until, self.clock() + completion.exhausted_for)
            return replace(completion, trail=tuple(trail))
        raise RouterExhausted(f"gave up after {attempts} attempts", trail)

    def _ready_in(self, endpoint: Endpoint, now: float) -> float:
        """Seconds until ``endpoint`` may be called (<= 0 means now)."""
        state = self._states[endpoint.label]
        wait = state.cooldown_until - now
        if endpoint.rpm:
            while state.calls and now - state.calls[0] >= 60.0:
                state.calls.popleft()
            if len(state.calls) >= endpoint.rpm:
                wait = max(wait, state.calls[0] + 60.0 - now)
        return wait

    def _same_provider(self, endpoint: Endpoint) -> list[Endpoint]:
        return [other for other in self.endpoints if other.provider.identifier == endpoint.provider.identifier]

    def _penalize(self, endpoint: Endpoint, error: ProviderError, skipped: set[str]) -> None:
        state = self._states[endpoint.label]
        now = self.clock()
        if error.kind == "rate_limited":
            state.strikes += 1
            backoff = min(MAXIMUM_COOLDOWN, RATE_LIMIT_BACKOFF * 2 ** (state.strikes - 1))
            delay = max(1.0, error.retry_after) if error.retry_after is not None else backoff
            targets = self._same_provider(endpoint) if endpoint.provider.shared_limit else [endpoint]
            for target in targets:
                target_state = self._states[target.label]
                target_state.cooldown_until = max(target_state.cooldown_until, now + delay)
        elif error.kind == "transient":
            delay = max(self.transient_cooldown, error.retry_after or 0.0)
            state.cooldown_until = max(state.cooldown_until, now + delay)
        elif error.kind == "unauthorized":
            for target in self._same_provider(endpoint):
                self._states[target.label].disabled = f"unauthorized: {error.message}"
        elif error.kind == "unavailable":
            state.disabled = error.message
        else:
            skipped.add(endpoint.label)

    def status(self) -> list[dict[str, object]]:
        """Per-endpoint health, for reports: disabled reason and remaining cool-down."""
        now = self.clock()
        return [
            {
                "endpoint": endpoint.label,
                "disabled": self._states[endpoint.label].disabled,
                "cooling_for": round(max(0.0, self._states[endpoint.label].cooldown_until - now), 1),
            }
            for endpoint in self.endpoints
        ]


def live_rosters(environment: Mapping[str, str], client: ChatClient | None = None) -> dict[str, list[str]]:
    """``GET /models`` of every configured live-roster provider (an unreachable one lists nothing)."""
    client = client or ChatClient()
    rosters: dict[str, list[str]] = {}
    for provider in PROVIDERS.values():
        endpoint = resolve_endpoint(provider, "", environment) if provider.live_roster else None
        if endpoint is None:
            continue
        try:
            rosters[provider.identifier] = client.list_models(endpoint)
        except ProviderError as error:
            logger.info("%s model list unavailable: %s", provider.identifier, error.message)
            rosters[provider.identifier] = []
    return rosters


def connect(
    specification: str = "auto",
    *,
    require_tools: bool = False,
    limit: int | None = None,
    catalog: Sequence[FreeModel] | None = None,
    environment: Mapping[str, str] | None = None,
    discover: bool = True,
    **options,
) -> ModelRouter:
    """Router over the configured free endpoints matching ``specification``.

    Reads ``.env`` (see :mod:`laboratory.agents.environment`) unless an
    ``environment`` mapping is given. ``specification`` is ``auto``, a model
    family, a provider or ``provider:model`` (comma-separate several; see
    :func:`~laboratory.agents.catalog.select_endpoints`). ``limit`` keeps the
    first ``limit`` endpoints of the failover chain. ``discover`` asks
    live-roster providers (AI Horde) which models are online right now.
    """
    if environment is None:
        load_environment()
        environment = os.environ
    rows = load_catalog() if catalog is None else catalog
    parts = [part.strip() for part in specification.split(",") if part.strip()]
    explicit = all(":" in part and part.partition(":")[0] in PROVIDERS for part in parts)
    rosters = live_rosters(environment) if discover and not explicit else {}
    endpoints = select_endpoints(specification, rows, environment, require_tools=require_tools, rosters=rosters)
    if limit is not None:
        endpoints = endpoints[:limit]
    if not endpoints:
        keyless = ", ".join(provider.identifier for provider in CATALOG_PROVIDERS.values() if provider.keyless)
        capability = " with tool calling" if require_tools else ""
        raise RouterExhausted(
            f"no configured free model matches {specification!r}{capability}. Keyless providers "
            f"({keyless}) need no key; for the others copy .env.example to .env and set <PROVIDER>_API_KEY."
        )
    return ModelRouter(endpoints, **options)
