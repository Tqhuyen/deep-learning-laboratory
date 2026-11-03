from __future__ import annotations

from pathlib import Path

from laboratory.agents.catalog import load_catalog
from laboratory.agents.environment import parse_environment_file
from laboratory.agents.providers import (
    CATALOG_PROVIDERS,
    PROVIDERS,
    configured_providers,
    resolve_endpoint,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_registry_and_snapshot_cover_the_same_providers():
    """A refreshed catalog with a new provider must fail here until the provider is registered."""
    in_snapshot = {row.provider for row in load_catalog()}
    assert in_snapshot == set(CATALOG_PROVIDERS)


def test_env_example_lists_every_variable_a_provider_reads():
    listed = parse_environment_file((REPOSITORY_ROOT / ".env.example").read_text(encoding="utf-8"))
    for provider in PROVIDERS.values():
        assert provider.key_variable in listed, provider.identifier
        for name in provider.required_variables():
            assert name in listed, (provider.identifier, name)


def test_keyless_providers_send_no_credentials_but_ai_horde_sends_its_anonymous_key():
    kilo = resolve_endpoint("kilo", "kilo-auto/free", {})
    horde = resolve_endpoint("aihorde", "koboldcpp/x", {})
    assert kilo is not None and "Authorization" not in kilo.headers()
    assert horde is not None and horde.headers()["Authorization"] == "Bearer 0000000000"


def test_keyed_provider_needs_its_key_and_never_prints_it():
    assert resolve_endpoint("groq", "llama", {}) is None
    endpoint = resolve_endpoint("groq", "llama", {"GROQ_API_KEY": "gsk_secret"})
    assert endpoint is not None
    assert endpoint.headers()["Authorization"] == "Bearer gsk_secret"
    assert "gsk_secret" not in repr(endpoint)


def test_url_placeholders_and_base_url_overrides_come_from_the_environment():
    environment = {"CLOUDFLARE_API_KEY": "token"}
    assert resolve_endpoint("cloudflare", "@cf/x", environment) is None
    endpoint = resolve_endpoint("cloudflare", "@cf/x", {**environment, "CLOUDFLARE_ACCOUNT_ID": "abc123"})
    assert endpoint.base_url == "https://api.cloudflare.com/client/v4/accounts/abc123/ai/v1"

    china = resolve_endpoint("zhipu", "glm", {"ZHIPU_API_KEY": "k", "ZHIPU_BASE_URL": "https://open.bigmodel.cn/api/paas/v4/"})
    assert china.base_url == "https://open.bigmodel.cn/api/paas/v4"


def test_custom_endpoint_needs_only_a_base_url():
    assert resolve_endpoint("custom", "llama3.2", {}) is None
    endpoint = resolve_endpoint("custom", "llama3.2", {"CUSTOM_BASE_URL": "http://localhost:11434/v1"})
    assert endpoint.label == "custom:llama3.2"


def test_provider_level_tool_support_overrides_the_catalog():
    assert resolve_endpoint("aihorde", "m", {}, tools=True).tools is False


def test_an_empty_environment_configures_exactly_the_keyless_catalog_providers():
    configured = {provider.identifier for provider in configured_providers({}, CATALOG_PROVIDERS.values())}
    assert configured == {identifier for identifier, provider in CATALOG_PROVIDERS.items() if provider.keyless}
