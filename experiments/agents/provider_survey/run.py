"""Which free models answer for *your* keys, how fast, and do they call tools?

Run this first after editing ``.env``. For every configured provider it takes
up to ``models_per_provider`` free models from ``data/free_models.json``
(tool-capable and strongest first; AI Horde from its live roster) and sends
two probes, each through a one-endpoint router so per-minute pacing applies:

* **echo** — "Reply with exactly: pong": does it answer, which model served
  the request, how long it took;
* **tool call** — a calculator tool plus "What is 1234 * 5678?": does the
  model return a structured call whose arguments evaluate to 7006652?
  (skipped for models the catalog lists without tool support).

All echo probes run before the tool probes, so a model's two requests are a
whole pass apart and 1-request-per-minute free tiers are not tripped.

    python experiments/agents/provider_survey/run.py --tier smoke
    python experiments/agents/provider_survey/run.py --tier development --providers groq,ovh,kilo
    python experiments/agents/provider_survey/run.py --tier smoke --models custom:llama3.2,freellmapi:auto

``tiny`` runs the identical harness on scripted offline endpoints. ``--steps``
overrides ``models_per_provider``. Artifacts: ``config.json``,
``metrics.json`` (one row per endpoint, a per-provider summary) and
``transcripts.jsonl`` (every request and reply).
"""

from __future__ import annotations

import json
import logging
import os
import platform
import statistics
from pathlib import Path

from laboratory.agents import (
    CATALOG_PROVIDERS,
    ChatModel,
    Message,
    ModelRouter,
    RouterExhausted,
    ScriptedModel,
    Toolbox,
    calculator,
    configured_providers,
    load_catalog,
    load_environment,
    select_endpoints,
    tool_call_message,
)
from laboratory.agents.routing import live_rosters
from laboratory.experiment import argument_parser, prepare_run, select_preset, write_config, write_metrics

AREA = "agents"
NAME = "provider_survey"

PRESETS = {
    "tiny": dict(offline=True, models_per_provider=2, max_tokens=256, max_wait=0.0),
    "smoke": dict(offline=False, models_per_provider=1, max_tokens=512, max_wait=70.0),
    "development": dict(offline=False, models_per_provider=3, max_tokens=512, max_wait=70.0),
    "full": dict(offline=False, models_per_provider=None, max_tokens=512, max_wait=70.0),
}

ECHO_PROMPT = "Reply with exactly: pong"
TOOL_PROMPT = "What is 1234 * 5678? Use the calculator tool."
TOOL_ANSWER = str(1234 * 5678)


def offline_targets() -> list[tuple[str, ChatModel, bool]]:
    """Scripted stand-ins: one model that passes both probes, one that answers in prose."""
    capable = ScriptedModel(["pong", tool_call_message("calculator", {"expression": "1234 * 5678"})], name="capable")
    prose = ScriptedModel(["Pong!", "1234 * 5678 = 7006652"], name="prose-only")
    return [("scripted:capable", capable, True), ("scripted:prose-only", prose, True)]


def live_targets(config: dict, args, environment) -> list[tuple[str, ChatModel, bool]]:
    if args.models:
        endpoints = select_endpoints(args.models, load_catalog(), environment, rosters=live_rosters(environment))
    else:
        wanted = set(args.providers.split(",")) if args.providers else set(CATALOG_PROVIDERS)
        providers = [p for p in configured_providers(environment, CATALOG_PROVIDERS.values()) if p.identifier in wanted]
        catalog, rosters = load_catalog(), live_rosters(environment)
        endpoints = []
        for provider in providers:
            candidates = select_endpoints(provider.identifier, catalog, environment, rosters=rosters)
            candidates.sort(key=lambda endpoint: not endpoint.tools)  # stable: rank order within each group
            endpoints += candidates[: config["models_per_provider"]]
    return [
        (endpoint.label, ModelRouter([endpoint], max_attempts=1, max_wait=config["max_wait"]), endpoint.tools)
        for endpoint in endpoints
    ]


def probe(model: ChatModel, messages: list[Message], toolbox: Toolbox | None, max_tokens: int) -> dict:
    tools = toolbox.tools if toolbox else ()
    try:
        completion = model.complete(messages, tools=tools, max_tokens=max_tokens)
    except (RouterExhausted, RuntimeError) as error:
        trail = getattr(error, "trail", ()) or (str(error),)
        return {"answered": False, "error": trail[-1][:300]}
    return {
        "answered": True,
        "text": completion.text.strip()[:200],
        "tool_calls": [{"name": call.name, "arguments": call.arguments} for call in completion.message.tool_calls],
        "served_model": completion.served_model,
        "latency": round(completion.latency, 2),
        "prompt_tokens": completion.usage.prompt_tokens,
        "completion_tokens": completion.usage.completion_tokens,
    }


def summarize(rows: list[dict]) -> dict:
    providers: dict[str, dict] = {}
    for row in rows:
        summary = providers.setdefault(row["provider"], {"endpoints": 0, "answered": 0, "followed": 0, "tool_ok": 0, "latencies": []})
        summary["endpoints"] += 1
        summary["answered"] += row["answered"]
        summary["followed"] += row["followed"]
        summary["tool_ok"] += bool(row["tool_ok"])
        if row["latency"] is not None:
            summary["latencies"].append(row["latency"])
    for summary in providers.values():
        latencies = summary.pop("latencies")
        summary["median_latency"] = round(statistics.median(latencies), 2) if latencies else None
    return providers


def main() -> None:
    parser = argument_parser(__doc__)
    parser.add_argument("--providers", default=None, help="comma-separated provider ids to survey (default: all configured)")
    parser.add_argument("--models", default=None, help="explicit model specification instead, e.g. custom:llama3.2")
    args = parser.parse_args()
    config = select_preset(PRESETS, args.tier)
    if args.steps is not None:
        config["models_per_provider"] = args.steps
    logging.basicConfig(level=logging.INFO, format="  %(message)s")

    if config["offline"]:
        targets = offline_targets()
        configured: list[str] = ["scripted"]
    else:
        load_environment()
        targets = live_targets(config, args, os.environ)
        configured = [provider.identifier for provider in configured_providers(os.environ)]
    print(f"python={platform.python_version()} tier={args.tier} configured providers: {', '.join(configured) or 'none'}")
    print(f"surveying {len(targets)} endpoints")

    run_directory = prepare_run(AREA, NAME, args.tier, args.output_directory)
    write_config(
        run_directory,
        {**config, "tier": args.tier, "seed": args.seed, "providers": args.providers, "models": args.models,
         "configured_providers": configured, "endpoints": [label for label, _, _ in targets]},
    )

    toolbox = Toolbox([calculator()])
    transcripts: list[dict] = []
    echoes: dict[str, dict] = {}
    for label, model, _ in targets:
        echoes[label] = probe(model, [Message.user(ECHO_PROMPT)], None, config["max_tokens"])
        transcripts.append({"endpoint": label, "probe": "echo", "prompt": ECHO_PROMPT, **echoes[label]})
        print(f"  echo  {label}: {'ok' if echoes[label]['answered'] else echoes[label]['error'][:120]}")

    rows = []
    for label, model, supports_tools in targets:
        echo = echoes[label]
        tool = None
        if echo["answered"] and supports_tools:
            tool = probe(model, [Message.user(TOOL_PROMPT)], toolbox, config["max_tokens"])
            transcripts.append({"endpoint": label, "probe": "tool_call", "prompt": TOOL_PROMPT, **tool})
        tool_ok = None
        if tool is not None:
            calls = [call for call in tool.get("tool_calls", []) if call["name"] == "calculator"]
            tool_ok = bool(calls) and toolbox.run("calculator", calls[0]["arguments"]) == TOOL_ANSWER
        provider, _, model_id = label.partition(":")
        rows.append(
            {
                "endpoint": label,
                "provider": provider,
                "model": model_id,
                "answered": echo["answered"],
                "followed": echo["answered"] and "pong" in echo.get("text", "").lower(),
                "served_model": echo.get("served_model"),
                "latency": echo.get("latency"),
                "tools_listed": supports_tools,
                "tool_ok": tool_ok,
                "error": echo.get("error") or (tool or {}).get("error"),
            }
        )

    print(f"\n{'endpoint':60s} answer pong  tools latency")
    for row in rows:
        tool_cell = "-" if row["tool_ok"] is None else ("yes" if row["tool_ok"] else "no")
        latency = f"{row['latency']:.1f}s" if row["latency"] is not None else "-"
        print(f"{row['endpoint'][:60]:60s} {'yes' if row['answered'] else 'no':6s} {'yes' if row['followed'] else 'no':5s} {tool_cell:5s} {latency}")

    summary = summarize(rows)
    working = [row["endpoint"] for row in rows if row["answered"]]
    tool_ready = [row["endpoint"] for row in rows if row["tool_ok"]]
    print(f"\n{len(working)}/{len(rows)} endpoints answered; {len(tool_ready)} made a correct tool call")
    with (Path(run_directory) / "transcripts.jsonl").open("w", encoding="utf-8") as handle:
        for record in transcripts:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_metrics(run_directory, rows, providers=summary, answered=working, tool_ready=tool_ready)
    print(f"artifacts: {run_directory}")


if __name__ == "__main__":
    main()
