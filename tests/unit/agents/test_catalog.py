from __future__ import annotations

import pytest

from laboratory.agents.catalog import (
    FreeModel,
    catalog_json,
    family_key,
    load_catalog,
    merge_catalogs,
    parse_freellmapi,
    parse_unorouter,
    select_endpoints,
)


def row(provider, model, *, family=None, rank=None, rpm=None, tools=True, guard=False):
    return FreeModel(provider, model, model, family or family_key(model), rank=rank, rpm=rpm, tools=tools, guard=guard)


CATALOG = [
    row("ovh", "gpt-oss-120b", rank=30, rpm=2),
    row("groq", "openai/gpt-oss-120b", family="gpt-oss-120b", rank=30, rpm=30),
    row("kilo", "big:free", rank=10),
    row("kilo", "guard:free", rank=1, guard=True),
    row("kilo", "small:free", rank=90, tools=False),
    row("llm7", "default", tools=False),
    row("aihorde", "stale-model", tools=False),
]


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("DeepSeek V4 Flash (Kilo)", "deepseek-v4-flash"),
        ("deepseek-v4-flash:free", "deepseek-v4-flash"),
        ("Qwen3.8 27B", "qwen3-8-27b"),
    ],
)
def test_family_keys_match_across_catalogs(name, key):
    assert family_key(name) == key


def test_freellmapi_rows_keep_only_enabled_chat_models():
    payload = {
        "models": [
            {"platform": "groq", "modelId": "a", "displayName": "A (Groq)", "supportsTools": True,
             "limits": {"rpm": 30, "rpd": 1000}, "intelligenceRank": 5},
            {"platform": "groq", "modelId": "tts", "displayName": "TTS", "modality": "audio"},
            {"platform": "groq", "modelId": "off", "displayName": "Off", "enabled": False},
            {"platform": "kilo", "modelId": "guard", "displayName": "Guard",
             "quirks": [{"slug": "safety-classifier-output", "title": "Safety classifier output"}]},
        ]
    }
    rows = parse_freellmapi(payload)
    assert [(r.model, r.name, r.rpm, r.rpd, r.rank, r.tools, r.guard) for r in rows] == [
        ("a", "A", 30, 1000, 5, True, False),
        ("guard", "Guard", None, None, None, False, True),
    ]


def test_unorouter_rows_are_free_online_chat_models_at_one_request_per_minute():
    base = {"is_free": True, "type": "text", "supported_endpoint_types": ["openai"], "online": True}
    payload = {
        "models": [
            {**base, "model_name": "glm-4.5-flash:free", "metadata": {"supportsTools": True, "contextWindow": 128000},
             "own_purpose": {"consent_required": True}},
            {**base, "model_name": "paid", "is_free": False},
            {**base, "model_name": "embed:free", "type": "embedding"},
            {**base, "model_name": "down:free", "online": False},
        ]
    }
    rows = parse_unorouter(payload)
    assert [(r.model, r.family, r.rpm, r.tools, r.context) for r in rows] == [
        ("glm-4.5-flash:free", "glm-4-5-flash", 1, True, 128000)
    ]
    assert rows[0].notes


def test_unorouter_list_replaces_freellmapi_unorouter_rows_and_inherits_family_rank():
    freellmapi = [row("unorouter", "stale:free", rank=3), row("groq", "glm-4.5-flash", rank=12)]
    unorouter = [row("unorouter", "glm-4.5-flash:free")]
    merged = merge_catalogs(freellmapi, unorouter)
    assert [(r.provider, r.model, r.rank) for r in merged] == [("groq", "glm-4.5-flash", 12), ("unorouter", "glm-4.5-flash:free", 12)]


def test_snapshot_round_trip(tmp_path):
    path = tmp_path / "free_models.json"
    path.write_text(catalog_json(CATALOG, [{"name": "test"}], "2026-01-01T00:00:00+00:00"), encoding="utf-8")
    assert load_catalog(path) == CATALOG


def labels(endpoints):
    return [endpoint.label for endpoint in endpoints]


def test_auto_uses_every_configured_chat_model_strongest_first():
    # Empty environment: only keyless providers; the guard model and the stale
    # live-roster provider (no roster fetched) never appear.
    assert labels(select_endpoints("auto", CATALOG, {})) == ["kilo:big:free", "ovh:gpt-oss-120b", "kilo:small:free", "llm7:default"]


def test_a_family_spans_providers_and_prefers_higher_rate_limits():
    environment = {"GROQ_API_KEY": "k"}
    assert labels(select_endpoints("gpt-oss-120b", CATALOG, environment)) == ["groq:openai/gpt-oss-120b", "ovh:gpt-oss-120b"]


def test_requiring_tools_filters_catalog_rows():
    assert labels(select_endpoints("kilo", CATALOG, {}, require_tools=True)) == ["kilo:big:free"]


def test_explicit_endpoints_may_be_outside_the_catalog_but_must_be_configured():
    environment = {"CUSTOM_BASE_URL": "http://localhost:11434/v1"}
    assert labels(select_endpoints("custom:llama3.2, kilo:big:free", CATALOG, environment)) == ["custom:llama3.2", "kilo:big:free"]
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        select_endpoints("groq:openai/gpt-oss-120b", CATALOG, {})
    with pytest.raises(ValueError, match="tool calling"):
        select_endpoints("llm7:default", CATALOG, {}, require_tools=True)


def test_live_rosters_replace_rotating_catalog_rows():
    endpoints = select_endpoints("aihorde", CATALOG, {}, rosters={"aihorde": ["koboldcpp/online"]})
    assert labels(endpoints) == ["aihorde:koboldcpp/online"]
    assert endpoints[0].tools is False
