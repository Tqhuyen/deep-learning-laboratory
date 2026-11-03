"""Which ``(provider, model)`` pairs cost $0, what they can do, and how to pick them.

``data/free_models.json`` is a committed snapshot of two public catalogs:

* FreeLLMAPI (``https://api.freellmapi.co/v1/latest``): every free chat model
  of 37 providers, with RPM/RPD limits, context window and tool support;
* UnoRouter (``https://unorouter.com/api/models/pricing/browse``): every
  ``:free`` text model, which replaces FreeLLMAPI's partial UnoRouter rows.

Refresh it with ``python data/prepare_free_models.py``. Free tiers change
weekly; a stale row costs one failed request, after which the router drops it.

Models are grouped into **families** (``deepseek-v4-flash``) with the same
normalisation freellmapi.co uses, so one family can be served by several
providers and the router can fail over without changing the weights.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .providers import PROVIDERS, Endpoint, resolve_endpoint

UNKNOWN_RANK = 1000
UNKNOWN_RPM = 10
"""Ordering assumptions for rows the catalogs leave blank."""


@dataclass(frozen=True)
class FreeModel:
    provider: str
    model: str
    name: str
    family: str
    context: int | None = None
    tools: bool = False
    vision: bool = False
    rank: int | None = None
    """FreeLLMAPI intelligence rank: 1 is the strongest free model."""
    rpm: int | None = None
    rpd: int | None = None
    guard: bool = False
    """A safety classifier that answers with labels, not a chat model."""
    notes: tuple[str, ...] = ()
    source: str = ""


def family_key(name: str) -> str:
    """``"DeepSeek V4 Flash (Kilo)"`` and ``"deepseek-v4-flash:free"`` -> ``"deepseek-v4-flash"``."""
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip().removesuffix(":free")
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip().replace(" ", "-")


def catalog_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "free_models.json"


# --- building the snapshot (used by data/prepare_free_models.py) ----------------


def parse_freellmapi(payload: Mapping[str, Any]) -> list[FreeModel]:
    """Chat rows of the FreeLLMAPI catalog (image, audio and TTS rows are dropped)."""
    rows = []
    for row in payload.get("models", []):
        if not row.get("enabled", True) or row.get("modality") not in (None, "chat", "text"):
            continue
        display = row.get("displayName") or row["modelId"]
        quirks = row.get("quirks") or []
        limits = row.get("limits") or {}
        rows.append(
            FreeModel(
                provider=row["platform"],
                model=row["modelId"],
                name=re.sub(r"\s*\([^)]*\)\s*$", "", display),
                family=family_key(display),
                context=row.get("contextWindow"),
                tools=bool(row.get("supportsTools")),
                vision=bool(row.get("supportsVision")),
                rank=row.get("intelligenceRank"),
                rpm=limits.get("rpm"),
                rpd=limits.get("rpd"),
                guard=any(quirk.get("slug") == "safety-classifier-output" for quirk in quirks),
                notes=tuple(quirk["title"] for quirk in quirks if quirk.get("title")),
                source="freellmapi",
            )
        )
    return rows


def parse_unorouter(payload: Mapping[str, Any]) -> list[FreeModel]:
    """``:free`` chat models of UnoRouter; its documented cap is 1 request/min per model."""
    rows = []
    for row in payload.get("models", []):
        endpoint_types = row.get("supported_endpoint_types") or []
        if not row.get("is_free") or row.get("type") != "text" or "openai" not in endpoint_types:
            continue
        if row.get("online") is False or row.get("chat") is False:
            continue
        metadata = row.get("metadata") or {}
        purpose = row.get("own_purpose") or {}
        notes = []
        if purpose.get("consent_required"):
            notes.append("Accept the provider consent on unorouter.com first")
        if purpose.get("trains_or_retains"):
            notes.append("Provider may train on or retain prompts")
        name = row["model_name"]
        rows.append(
            FreeModel(
                provider="unorouter",
                model=name,
                name=name.removesuffix(":free"),
                family=family_key(name),
                context=metadata.get("contextWindow"),
                tools=bool(metadata.get("supportsTools")),
                vision=bool(metadata.get("supportsVision")),
                rpm=1,
                guard=bool(re.search(r"guard|content-safety", name)),
                notes=tuple(notes),
                source="unorouter",
            )
        )
    return rows


def merge_catalogs(freellmapi: Sequence[FreeModel], unorouter: Sequence[FreeModel]) -> list[FreeModel]:
    """UnoRouter's own list replaces FreeLLMAPI's UnoRouter rows; ranks carry over by family."""
    best_rank: dict[str, int] = {}
    for row in freellmapi:
        if row.rank is not None:
            best_rank[row.family] = min(best_rank.get(row.family, row.rank), row.rank)
    rows = [row for row in freellmapi if not (unorouter and row.provider == "unorouter")]
    rows += [replace(row, rank=row.rank if row.rank is not None else best_rank.get(row.family)) for row in unorouter]
    unique = {(row.provider, row.model): row for row in rows}
    return sorted(unique.values(), key=lambda row: (row.provider, row.model.lower()))


def catalog_json(models: Iterable[FreeModel], sources: Sequence[Mapping[str, Any]], generated_at: str) -> str:
    """Snapshot text with one model per line, so refreshes produce readable diffs."""
    lines = [json.dumps({**asdict(model), "notes": list(model.notes)}, ensure_ascii=False) for model in models]
    header = json.dumps({"generated_at": generated_at, "sources": list(sources)}, indent=2, ensure_ascii=False)
    return header[:-2] + ',\n  "models": [\n    ' + ",\n    ".join(lines) + "\n  ]\n}\n"


# --- using the snapshot -----------------------------------------------------------


def load_catalog(path: Path | str | None = None) -> list[FreeModel]:
    path = Path(path) if path is not None else catalog_path()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [FreeModel(**{**record, "notes": tuple(record.get("notes", ()))}) for record in payload["models"]]


def _order(row: FreeModel) -> tuple[int, int]:
    rank = row.rank if row.rank is not None else UNKNOWN_RANK
    rpm = row.rpm if row.rpm is not None else UNKNOWN_RPM
    return rank, -rpm


def _endpoint(row: FreeModel, environment: Mapping[str, str]) -> Endpoint | None:
    return resolve_endpoint(
        row.provider, row.model, environment, tools=row.tools, family=row.family, rpm=row.rpm, context=row.context
    )


def _select_one(
    part: str, catalog: Sequence[FreeModel], environment: Mapping[str, str], require_tools: bool
) -> list[Endpoint]:
    provider, separator, model = part.partition(":")
    if separator and provider in PROVIDERS:
        row = next((r for r in catalog if r.provider == provider and r.model == model), None)
        if row is None:
            row = FreeModel(provider, model, model, family_key(model), tools=True)
        endpoint = _endpoint(row, environment)
        if endpoint is None:
            needed = ", ".join(PROVIDERS[provider].required_variables())
            raise ValueError(f"{part!r} is not configured: set {needed} in .env")
        if require_tools and not endpoint.tools:
            raise ValueError(f"{part!r} does not support tool calling")
        return [endpoint]

    rows = [row for row in catalog if row.provider in PROVIDERS and not row.guard]
    if part in PROVIDERS:
        rows = [row for row in rows if row.provider == part]
    elif part != "auto":
        rows = [row for row in rows if row.family == family_key(part)]
    if require_tools:
        rows = [row for row in rows if row.tools and PROVIDERS[row.provider].tools]
    endpoints = (_endpoint(row, environment) for row in sorted(rows, key=_order))
    return [endpoint for endpoint in endpoints if endpoint is not None]


def with_rosters(catalog: Sequence[FreeModel], rosters: Mapping[str, Sequence[str]]) -> list[FreeModel]:
    """Replace the catalog rows of each provider in ``rosters`` with its live model list."""
    rows = [row for row in catalog if row.provider not in rosters]
    for provider, models in rosters.items():
        known = {row.model: row for row in catalog if row.provider == provider}
        tools = PROVIDERS[provider].tools if provider in PROVIDERS else False
        rows += [known.get(model) or FreeModel(provider, model, model, family_key(model), tools=tools) for model in models]
    return rows


def select_endpoints(
    specification: str,
    catalog: Sequence[FreeModel],
    environment: Mapping[str, str],
    *,
    require_tools: bool = False,
    rosters: Mapping[str, Sequence[str]] | None = None,
) -> list[Endpoint]:
    """Configured endpoints for a comma-separated model specification, in failover order.

    Each part is one of:

    * ``auto`` — every configured free model, strongest first;
    * a family, e.g. ``gpt-oss-120b`` — that model on every configured provider;
    * a provider, e.g. ``groq`` — every free model of that provider;
    * ``provider:model``, e.g. ``ovh:gpt-oss-120b`` or ``custom:llama3.2`` — one
      endpoint, which may be outside the catalog.

    ``rosters`` maps a provider to its live model list (see
    :attr:`~laboratory.agents.providers.Provider.live_roster`); those lists
    replace the provider's catalog rows. Without one, live-roster providers are
    only used through explicit ``provider:model`` parts.
    """
    rosters = dict(rosters or {})
    stale = {identifier for identifier, provider in PROVIDERS.items() if provider.live_roster and identifier not in rosters}
    rows = [row for row in with_rosters(catalog, rosters) if row.provider not in stale]
    selected: dict[str, Endpoint] = {}
    for part in (piece.strip() for piece in specification.split(",")):
        if part:
            for endpoint in _select_one(part, rows, environment, require_tools):
                selected.setdefault(endpoint.label, endpoint)
    return list(selected.values())
