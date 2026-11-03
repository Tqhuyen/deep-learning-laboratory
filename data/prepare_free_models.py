"""Refresh ``data/free_models.json``, the snapshot of every $0 chat model.

Usage:

    python data/prepare_free_models.py

Sources (both public, no key needed):

* FreeLLMAPI catalog — https://api.freellmapi.co/v1/latest, the feed behind
  https://freellmapi.co/models (37 chat providers, per-model limits);
* UnoRouter model browser — https://unorouter.com/api/models/pricing/browse,
  the data behind https://unorouter.com/en/models?max_price=0.

Free tiers change weekly, so re-run this before a long experiment. Agent runs
never download anything themselves; this script is the explicit preparation
step required by ``docs/conventions.md``. If one source is unreachable the
previous rows for it are kept.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laboratory.agents.catalog import (
    catalog_json,
    catalog_path,
    load_catalog,
    merge_catalogs,
    parse_freellmapi,
    parse_unorouter,
)
from laboratory.agents.providers import CATALOG_PROVIDERS, USER_AGENT

FREELLMAPI_URL = "https://api.freellmapi.co/v1/latest"
UNOROUTER_URL = "https://unorouter.com/api/models/pricing/browse"


def fetch(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--destination", type=Path, default=catalog_path(), help="snapshot to write")
    args = parser.parse_args()

    previous = load_catalog(args.destination) if args.destination.exists() else []
    sources = []
    try:
        freellmapi_payload = fetch(FREELLMAPI_URL)
        freellmapi = parse_freellmapi(freellmapi_payload)
        sources.append({"name": "freellmapi", "url": FREELLMAPI_URL, "version": freellmapi_payload.get("version")})
    except (OSError, ValueError) as error:
        print(f"FreeLLMAPI catalog unavailable ({error}); keeping the previous rows")
        freellmapi = [row for row in previous if row.source == "freellmapi"]
    try:
        unorouter = parse_unorouter(fetch(UNOROUTER_URL))
        sources.append({"name": "unorouter", "url": UNOROUTER_URL})
    except (OSError, ValueError) as error:
        print(f"UnoRouter catalog unavailable ({error}); keeping the previous rows")
        unorouter = [row for row in previous if row.source == "unorouter"]

    models = merge_catalogs(freellmapi, unorouter)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    args.destination.write_text(catalog_json(models, sources, generated_at), encoding="utf-8")

    per_provider = Counter(model.provider for model in models)
    unknown = sorted(set(per_provider) - set(CATALOG_PROVIDERS))
    print(f"wrote {args.destination}: {len(models)} chat models from {len(per_provider)} providers")
    print(f"  with tool calling: {sum(model.tools for model in models)}")
    print("  " + ", ".join(f"{provider}={count}" for provider, count in sorted(per_provider.items())))
    if unknown:
        print(f"  not in laboratory/agents/providers.py yet (skipped by the router): {', '.join(unknown)}")


if __name__ == "__main__":
    main()
