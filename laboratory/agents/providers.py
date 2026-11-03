"""Every free chat provider in the FreeLLMAPI and UnoRouter catalogs.

A :class:`Provider` is the *static* description of one HTTP API: base URL,
where to get a key, and the ways it diverges from the OpenAI Chat Completions
contract. An :class:`Endpoint` is one callable ``(provider, model)`` pair with
the URL and key resolved from the environment.

Environment variables follow one rule, so nothing has to be looked up::

    <ID>_API_KEY    the key, e.g. GROQ_API_KEY, UNOROUTER_API_KEY
    <ID>_BASE_URL   optional base-URL override, e.g. ZHIPU_BASE_URL

Cloudflare additionally needs ``CLOUDFLARE_ACCOUNT_ID`` (it is part of the
URL). Keyless providers (Kilo, OVH, LLM7, AI Horde) answer anonymously when
their key is unset; a key only raises their limits.

The quirk fields mirror the adapters of the FreeLLMAPI router
(github.com/tashfeenahmed/freellmapi, ``server/src/providers``), where each
divergence was found against the live API.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from .environment import variable

USER_AGENT = (
    "Mozilla/5.0 (compatible; deep-learning-laboratory/0.1; "
    "+https://github.com/Tqhuyen/deep-learning-laboratory)"
)
"""Some free gateways sit behind Cloudflare rules that reject non-browser agents."""

_PLACEHOLDER = re.compile(r"\{([A-Z0-9_]+)\}")


@dataclass(frozen=True)
class Provider:
    """One OpenAI-compatible (or Responses-API) HTTP provider."""

    identifier: str
    name: str
    base_url: str
    signup_url: str
    notes: str = ""
    keyless: bool = False
    """Anonymous requests work when ``<ID>_API_KEY`` is unset."""
    anonymous_key: str | None = None
    """Bearer token sent for anonymous requests (AI Horde's documented ``0000000000``)."""
    wire: str = "chat"
    """``"chat"`` for ``/chat/completions``, ``"responses"`` for the OpenAI Responses API."""
    headers: tuple[tuple[str, str], ...] = ()
    timeout: float = 90.0
    tools: bool = True
    """False when the API rejects or silently drops ``tools``."""
    stop: bool = True
    """False when the API rejects stop sequences."""
    single_tool_call: bool = False
    """Send ``parallel_tool_calls: false`` whenever tools are present."""
    max_tokens_field: str = "max_tokens"
    minimum_max_tokens: int | None = None
    default_max_tokens: int | None = None
    strip_schema_keys: frozenset[str] = field(default_factory=frozenset)
    """JSON-schema keywords the tool validator rejects; removed recursively."""
    verify_model: bool = False
    """The response must name the requested model; free gateways substitute silently."""
    shared_limit: bool = False
    """A rate limit on one model applies to the whole account."""
    live_roster: bool = False
    """Free models rotate faster than any catalog; the live ``GET /models`` list is authoritative."""

    @property
    def key_variable(self) -> str:
        return f"{self.identifier.upper()}_API_KEY"

    @property
    def base_url_variable(self) -> str:
        return f"{self.identifier.upper()}_BASE_URL"

    def required_variables(self) -> list[str]:
        """Variables that must be set before this provider can be called."""
        required = _PLACEHOLDER.findall(self.base_url)
        if not self.base_url:
            required.append(self.base_url_variable)
        if not self.keyless:
            required.append(self.key_variable)
        return required


@dataclass(frozen=True)
class Endpoint:
    """A resolved ``(provider, model)`` pair: URL, credentials and catalog facts."""

    provider: Provider
    model: str
    base_url: str
    api_key: str | None = field(default=None, repr=False)
    tools: bool = True
    family: str = ""
    rpm: int | None = None
    context: int | None = None

    @property
    def label(self) -> str:
        return f"{self.provider.identifier}:{self.model}"

    def headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
            **dict(self.provider.headers),
        }
        token = self.api_key or self.provider.anonymous_key
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers


def _provider(*args, **kwargs) -> tuple[str, Provider]:
    provider = Provider(*args, **kwargs)
    return provider.identifier, provider


_CLOUDFLARE_URL = "https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1"
_SCHEMA_KEYS_GEMINI_AND_COHERE_REJECT = frozenset({"additionalProperties", "$schema"})

CATALOG_PROVIDERS: dict[str, Provider] = dict(
    [
        # --- no key needed -------------------------------------------------------
        _provider("kilo", "Kilo Gateway", "https://api.kilo.ai/api/gateway/v1", "https://app.kilo.ai",
                  ":free routes answer anonymously, 200 requests/hour per IP; free prompts may be logged",
                  keyless=True),
        _provider("ovh", "OVH AI Endpoints", "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
                  "https://endpoints.ai.cloud.ovh.net",
                  "anonymous tier: 2 requests/min per IP per model", keyless=True),
        _provider("llm7", "LLM7", "https://api.llm7.io/v1", "https://llm7.io",
                  "anonymous access works; a free token raises the limits", keyless=True),
        _provider("aihorde", "AI Horde", "https://oai.aihorde.net/v1", "https://aihorde.net/register",
                  "volunteer GPUs: slow queue, rotating roster, no tool calling; a key raises queue priority",
                  keyless=True, anonymous_key="0000000000", timeout=180.0, tools=False,
                  minimum_max_tokens=16, default_max_tokens=512, live_roster=True),
        # --- first-party labs ----------------------------------------------------
        _provider("google", "Google AI Studio", "https://generativelanguage.googleapis.com/v1beta/openai",
                  "https://aistudio.google.com/apikey",
                  "per-model RPM/RPD caps; hidden thinking tokens count against max_tokens",
                  strip_schema_keys=_SCHEMA_KEYS_GEMINI_AND_COHERE_REJECT),
        _provider("groq", "Groq", "https://api.groq.com/openai/v1", "https://console.groq.com/keys",
                  "free key, no card; per-model RPM/RPD caps"),
        _provider("mistral", "Mistral", "https://api.mistral.ai/v1", "https://console.mistral.ai/api-keys/",
                  "free Experiment plan"),
        _provider("cohere", "Cohere", "https://api.cohere.ai/compatibility/v1",
                  "https://dashboard.cohere.com/api-keys", "trial key with monthly call caps",
                  strip_schema_keys=_SCHEMA_KEYS_GEMINI_AND_COHERE_REJECT),
        _provider("nvidia", "NVIDIA NIM", "https://integrate.api.nvidia.com/v1",
                  "https://build.nvidia.com/settings/api-keys",
                  "about 40 RPM per account; evaluation-only terms; one tool call per turn",
                  timeout=180.0, single_tool_call=True),
        _provider("cloudflare", "Cloudflare Workers AI", _CLOUDFLARE_URL, "https://dash.cloudflare.com",
                  "10,000 neurons/day; API token with Workers AI permission plus CLOUDFLARE_ACCOUNT_ID",
                  timeout=200.0),
        _provider("zhipu", "Zhipu AI (Z.ai)", "https://api.z.ai/api/paas/v4",
                  "https://z.ai/manage-apikey/apikey-list",
                  "keys from open.bigmodel.cn need ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4"),
        _provider("modelscope", "ModelScope", "https://api-inference.modelscope.cn/v1",
                  "https://modelscope.cn/my/myaccesstoken",
                  "the token works only after binding a real-name Alibaba Cloud China account"),
        _provider("sealion", "SEA-LION", "https://api.sea-lion.ai/v1", "https://sea-lion.ai",
                  "AI Singapore; 10 RPM"),
        _provider("agnes", "Agnes AI", "https://apihub.agnes-ai.com/v1", "https://platform.agnes-ai.com",
                  "promotional $0 pricing on Agnes models"),
        _provider("radeon", "AMD Radeon Cloud", "https://developer.amd.com.cn/radeon/api/v1",
                  "https://developer.amd.com.cn/radeon/tokenfactory",
                  "rotating free roster; one tool call per turn; long reasoning",
                  timeout=600.0, single_tool_call=True),
        _provider("moondream", "Moondream", "https://api.moondream.ai/v1",
                  "https://moondream.ai/c/cloud/api-keys",
                  "vision model; $5/month shared credits; no tools or stop sequences",
                  tools=False, stop=False, max_tokens_field="max_completion_tokens", verify_model=True),
        # --- routers and gateways -------------------------------------------------
        _provider("unorouter", "UnoRouter", "https://api.unorouter.com/v1", "https://unorouter.com",
                  ":free models: 1 request/min per model; bursts trip an account-wide 429",
                  shared_limit=True),
        _provider("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "https://openrouter.ai/keys",
                  ":free models share 50 requests/day per account (1,000 after a one-time $10 top-up)",
                  headers=(("HTTP-Referer", "https://github.com/Tqhuyen/deep-learning-laboratory"),
                           ("X-Title", "deep-learning-laboratory")),
                  shared_limit=True),
        _provider("xkiro", "xKiro", "https://api.xkiro.com/v1", "https://xkiro.com",
                  "5M free tokens/day; premium models answer 403"),
        _provider("electronhub", "ElectronHub", "https://api.electronhub.ai/v1", "https://app.electronhub.ai",
                  "shared weekly free credits"),
        _provider("requesty", "Requesty", "https://router.requesty.ai/v1", "https://www.requesty.ai",
                  "free key, no card"),
        _provider("routeway", "Routeway", "https://api.routeway.ai/v1", "https://routeway.ai",
                  ":free models, roughly 5-20 RPM"),
        _provider("nara", "NaraRouter", "https://router.bynara.id/v1", "https://router.bynara.id",
                  "daily free quota after Telegram verification"),
        _provider("bazaarlink", "BazaarLink", "https://bazaarlink.ai/api/v1", "https://bazaarlink.ai",
                  "only the auto:free route is free"),
        _provider("aion", "Aion Labs", "https://api.aionlabs.ai/v1", "https://www.aionlabs.ai",
                  "free key, no card"),
        _provider("router9", "Router9", "https://api.router9.com/v1", "https://www.router9.com",
                  "shared monthly free credits"),
        _provider("speka", "Speka", "https://speka.me/v1", "https://speka.me/dashboard/keys",
                  "shared $1/month free credits"),
        _provider("septor", "Septor Labs", "https://api.septorlabs.com/v1", "https://septorlabs.com/dashboard",
                  "shared daily free quota; several aliases substitute another model", verify_model=True),
        _provider("clod", "CLōD", "https://api.clod.io/v1", "https://app.clod.io",
                  "shared daily free requests", verify_model=True),
        _provider("airforce", "Api.Airforce", "https://api.airforce/v1", "https://api.airforce",
                  "1 request/min per account", verify_model=True, shared_limit=True),
        _provider("dreamprompting", "DreamPrompting", "https://dreamprompting.com/api/v1",
                  "https://dreamprompting.com",
                  "500K tokens and 5,000 requests per rolling 24 h; pooled router", verify_model=True),
        _provider("waterfall", "Waterfall", "https://api.getwaterfall.org/v1", "https://getwaterfall.org",
                  "community free models", verify_model=True),
        _provider("logfare", "Logfare", "https://logfare.ai/v1", "https://logfare.ai",
                  "fair-use free tier that logs all traffic", verify_model=True),
        _provider("llmtr", "LLMTR", "https://llmtr.com/v1", "https://llmtr.com",
                  "daily free-model quotas; automated key probing gets accounts suspended", verify_model=True),
        _provider("gizmo", "Gizmo", "https://api.gizmoplatforms.com/api/v1",
                  "https://gizmoplatforms.com/developers", "monthly free requests; text only, no tools",
                  tools=False),
        _provider("blockrun", "BlockRun", "https://api.blockrun.ai/v1", "https://user.blockrun.ai/dashboard/keys",
                  "zero-priced routes", verify_model=True),
        _provider("aclide", "ACLIDE", "https://aclide.com/v1", "https://aclide.com/en/dashboard/api-keys",
                  "OpenAI Responses API; shared monthly credits; closed to new sign-ups",
                  wire="responses", timeout=120.0),
    ]
)
"""The 37 chat providers of the free catalogs (Speechify is TTS-only and absent)."""

LOCAL_PROVIDERS: dict[str, Provider] = dict(
    [
        _provider("freellmapi", "FreeLLMAPI router", "http://localhost:3001/v1",
                  "https://github.com/tashfeenahmed/freellmapi",
                  "self-hosted router over the providers above; models auto, auto:fast, auto:smart"),
        _provider("custom", "Custom OpenAI-compatible endpoint", "", "",
                  "Ollama http://localhost:11434/v1, LM Studio http://localhost:1234/v1, vLLM, llama.cpp",
                  keyless=True, timeout=300.0),
    ]
)

PROVIDERS: dict[str, Provider] = {**CATALOG_PROVIDERS, **LOCAL_PROVIDERS}


def resolve_base_url(provider: Provider, environment: Mapping[str, str]) -> str | None:
    """Base URL with ``{VARIABLE}`` placeholders filled, or None if one is unset."""
    base_url = variable(environment, provider.base_url_variable) or provider.base_url
    missing = [name for name in _PLACEHOLDER.findall(base_url) if variable(environment, name) is None]
    if not base_url or missing:
        return None
    filled = _PLACEHOLDER.sub(lambda match: variable(environment, match.group(1)) or "", base_url)
    return filled.rstrip("/")


def resolve_endpoint(
    provider: Provider | str,
    model: str,
    environment: Mapping[str, str],
    *,
    tools: bool = True,
    family: str = "",
    rpm: int | None = None,
    context: int | None = None,
) -> Endpoint | None:
    """Bind ``model`` on ``provider`` to URL and key; None when not configured."""
    provider = PROVIDERS[provider] if isinstance(provider, str) else provider
    base_url = resolve_base_url(provider, environment)
    api_key = variable(environment, provider.key_variable)
    if base_url is None or (api_key is None and not provider.keyless):
        return None
    return Endpoint(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        tools=tools and provider.tools,
        family=family,
        rpm=rpm,
        context=context,
    )


def configured_providers(
    environment: Mapping[str, str], providers: Iterable[Provider] | None = None
) -> list[Provider]:
    """Providers whose URL and credentials resolve in ``environment``."""
    candidates = PROVIDERS.values() if providers is None else providers
    return [provider for provider in candidates if resolve_endpoint(provider, "probe", environment)]
