"""Agent architectures on the same tasks, the same free model and the same tools.

Seven architectures from ``laboratory/agents`` answer the questions of
``data/agent_tasks.json``, a fictional world where arithmetic needs the
calculator and every fact needs the lookup tool:

==================  ==================  ==========================================
architecture        file                idea
==================  ==================  ==========================================
direct              loops.py            one call, no tools (the floor)
react               loops.py            Thought/Action/Observation as text (Yao 2022)
tool_calling        loops.py            native function calling
plan_and_execute    planning.py         plan first, execute steps (Wang 2023)
reflexion           reflection.py       self-evaluate, reflect, retry (Shinn 2023)
self_consistency    ensembles.py        majority vote over samples (Wang 2022)
debate              ensembles.py        agents revise after reading each other (Du 2023)
==================  ==================  ==========================================

Every architecture shares one router, restricted to tool-capable endpoints
whenever any selected architecture needs native tools, so all of them run on
the same endpoints. Pin a model **family** for a clean comparison: failover
then changes the provider, never the weights.

    python experiments/agents/architecture_comparison/run.py --tier smoke
    python experiments/agents/architecture_comparison/run.py --tier development --model gpt-oss-120b
    python experiments/agents/architecture_comparison/run.py --tier development --architectures react,reflexion --inner tool_calling

``tiny`` replays scripted model outputs (no network) through every
architecture's real control flow. ``--steps`` overrides ``max_steps``, the
action budget of each inner loop. Add an architecture by writing it in
``laboratory/agents/`` and registering a builder in ``BUILDERS`` below.
"""

from __future__ import annotations

import json
import logging
import platform
import statistics
from collections import Counter
from pathlib import Path

from laboratory.agents import (
    Agent,
    ChatModel,
    DebateAgent,
    DirectAgent,
    PlanAndExecuteAgent,
    ReActAgent,
    ReflexionAgent,
    ScriptedModel,
    SelfConsistencyAgent,
    Toolbox,
    ToolCallingAgent,
    answers_match,
    connect,
    load_task_suite,
    tool_call_message,
)
from laboratory.experiment import argument_parser, prepare_run, select_preset, write_config, write_metrics
from laboratory.utilities import set_seed

AREA = "agents"
NAME = "architecture_comparison"

ARCHITECTURES = ("direct", "react", "tool_calling", "plan_and_execute", "reflexion", "self_consistency", "debate")

_SHARED = dict(
    max_steps=6,
    inner="react",
    temperature=0.2,
    sample_temperature=0.7,
    max_tokens=2048,
    plan_steps=4,
    trials=2,
    samples=3,
    debaters=3,
    rounds=2,
    max_wait=180.0,
    endpoint_limit=None,
)
PRESETS = {
    "tiny": dict(_SHARED, offline=True, tasks=1, seeds=1, architectures=ARCHITECTURES, samples=2, debaters=2),
    "smoke": dict(_SHARED, offline=False, tasks=4, seeds=1, architectures=("direct", "react", "tool_calling")),
    "development": dict(_SHARED, offline=False, tasks=12, seeds=1, architectures=ARCHITECTURES),
    "full": dict(_SHARED, offline=False, tasks=24, seeds=3, architectures=ARCHITECTURES),
}


# --- architectures ----------------------------------------------------------------


def _inner(model: ChatModel, toolbox: Toolbox, config: dict, temperature: float) -> Agent:
    agent_class = ReActAgent if config["inner"] == "react" else ToolCallingAgent
    return agent_class(model, toolbox, max_steps=config["max_steps"], temperature=temperature, max_tokens=config["max_tokens"])


def _sampling(config: dict) -> dict:
    return {"temperature": config["temperature"], "max_tokens": config["max_tokens"]}


BUILDERS = {
    "direct": lambda model, toolbox, config: DirectAgent(model, **_sampling(config)),
    "react": lambda model, toolbox, config: ReActAgent(model, toolbox, max_steps=config["max_steps"], **_sampling(config)),
    "tool_calling": lambda model, toolbox, config: ToolCallingAgent(
        model, toolbox, max_steps=config["max_steps"], **_sampling(config)
    ),
    "plan_and_execute": lambda model, toolbox, config: PlanAndExecuteAgent(
        model, _inner(model, toolbox, config, config["temperature"]), toolbox,
        max_plan_steps=config["plan_steps"], **_sampling(config),
    ),
    "reflexion": lambda model, toolbox, config: ReflexionAgent(
        model, _inner(model, toolbox, config, config["temperature"]), trials=config["trials"], **_sampling(config)
    ),
    "self_consistency": lambda model, toolbox, config: SelfConsistencyAgent(
        _inner(model, toolbox, config, config["sample_temperature"]), samples=config["samples"]
    ),
    "debate": lambda model, toolbox, config: DebateAgent(
        _inner(model, toolbox, config, config["sample_temperature"]), debaters=config["debaters"], rounds=config["rounds"]
    ),
}
"""``name -> builder(model, toolbox, config) -> Agent``; register new architectures here."""


# --- the offline tier -------------------------------------------------------------

_REACT_STEP = 'Thought: I need the product.\nAction: calculator\nAction Input: {"expression": "48763 * 2917"}'
_CALL_STEP = tool_call_message("calculator", {"expression": "48763 * 2917"})


def _repeat_observation(messages, tools) -> str:
    """Scripted final turn: answer with the latest tool result."""
    for message in reversed(messages):
        if message.role == "tool":
            return f"Final Answer: {message.content}"
        if message.role == "user" and message.content.startswith("Observation:"):
            return f"Thought: I now know the answer\nFinal Answer: {message.content.removeprefix('Observation:').strip()}"
    return "Final Answer: unknown"


def _answer_from_results(messages, tools) -> str:
    return "Final Answer: " + messages[-1].content.rsplit("-> ", 1)[-1]


_REACT_ATTEMPT = [_REACT_STEP, _repeat_observation]
TINY_SCRIPTS = {
    "direct": ["48,763 x 2,917 is 142,241,671.\nFinal Answer: 142241671"],
    "react": _REACT_ATTEMPT,
    "tool_calling": [_CALL_STEP, _repeat_observation],
    "plan_and_execute": ["1. Multiply 48,763 by 2,917 with the calculator.", *_REACT_ATTEMPT, _answer_from_results],
    "reflexion": [
        *_REACT_ATTEMPT,
        "Verdict: INCORRECT\nReason: the product was not double-checked.",
        "Recompute the product with the calculator before answering.",
        *_REACT_ATTEMPT,
        "Verdict: CORRECT\nReason: the calculator result is the answer.",
    ],
    "self_consistency": _REACT_ATTEMPT * 2,
    "debate": _REACT_ATTEMPT * 4,
}
"""Model outputs for the ``tiny`` tier, which runs the first task (arithmetic-01)."""


# --- reporting --------------------------------------------------------------------


def summarize(records: list[dict], seeds: list[int]) -> dict:
    summary = {}
    for architecture in dict.fromkeys(record["architecture"] for record in records):
        rows = [record for record in records if record["architecture"] == architecture]
        by_seed = [statistics.mean(r["correct"] for r in rows if r["seed"] == seed) for seed in seeds]
        kinds = dict.fromkeys(r["kind"] for r in rows)
        summary[architecture] = {
            "accuracy": statistics.mean(by_seed),
            "accuracy_std": statistics.stdev(by_seed) if len(by_seed) > 1 else 0.0,
            "accuracy_by_seed": by_seed,
            "accuracy_by_kind": {kind: statistics.mean(r["correct"] for r in rows if r["kind"] == kind) for kind in kinds},
            "calls_per_task": statistics.mean(r["calls"] for r in rows),
            "tool_calls_per_task": statistics.mean(r["tool_calls"] for r in rows),
            "tokens_per_task": statistics.mean(r["prompt_tokens"] + r["completion_tokens"] for r in rows),
            "seconds_per_task": statistics.mean(r["latency"] for r in rows),
            "errors": sum(r["stopped"] == "error" for r in rows),
            "out_of_steps": sum(r["stopped"] == "max_steps" for r in rows),
        }
    return summary


def main() -> None:
    parser = argument_parser(__doc__)
    parser.add_argument("--model", default="auto", help="auto, a model family, a provider, or provider:model (comma-separated)")
    parser.add_argument("--architectures", default=None, help=f"comma-separated subset of {', '.join(ARCHITECTURES)}")
    parser.add_argument("--inner", choices=("react", "tool_calling"), default=None, help="inner agent of composite architectures")
    args = parser.parse_args()
    config = select_preset(PRESETS, args.tier)
    if args.steps is not None:
        config["max_steps"] = args.steps
    if args.inner is not None:
        config["inner"] = args.inner
    architectures = tuple(args.architectures.split(",")) if args.architectures else tuple(config["architectures"])
    unknown = sorted(set(architectures) - set(BUILDERS))
    if unknown:
        raise SystemExit(f"unknown architectures {unknown}; choose from {sorted(BUILDERS)}")
    logging.basicConfig(level=logging.INFO, format="    %(message)s")

    suite = load_task_suite()
    tasks = suite.tasks[: config["tasks"]]
    toolbox = suite.toolbox()
    seeds = list(range(args.seed, args.seed + config["seeds"]))

    router = None
    if not config["offline"]:
        probe_agents = [BUILDERS[name](None, toolbox, config) for name in architectures]
        router = connect(
            args.model,
            require_tools=any(agent.requires_tools for agent in probe_agents),
            limit=config["endpoint_limit"],
            max_wait=config["max_wait"],
        )
    endpoints = [endpoint.label for endpoint in router.endpoints] if router else ["scripted"]
    print(f"python={platform.python_version()} tier={args.tier} model={args.model!r}: {len(endpoints)} endpoint(s), first {endpoints[:3]}")

    run_directory = prepare_run(AREA, NAME, args.tier, args.output_directory)
    write_config(
        run_directory,
        {**config, "architectures": architectures, "tier": args.tier, "seeds": seeds, "model": args.model,
         "endpoints": endpoints, "tasks": [task.identifier for task in tasks]},
    )

    records, transcripts = [], []
    served: Counter[str] = Counter()
    for seed in seeds:
        set_seed(seed)
        for architecture in architectures:
            for task in tasks:
                model = ScriptedModel(TINY_SCRIPTS[architecture]) if router is None else router
                episode = BUILDERS[architecture](model, toolbox, config).run(task.question)
                correct = answers_match(episode.answer, task.answer)
                served.update(completion.endpoint for completion in episode.completions)
                transcript = {"seed": seed, "task": task.identifier, "kind": task.kind, "reference": task.answer,
                              "correct": correct, **episode.to_record()}
                transcripts.append(transcript)
                records.append({key: value for key, value in transcript.items() if key not in ("messages", "events", "completions")})
                verdict = "correct" if correct else ("error: " + (episode.error or "")[:120] if episode.stopped == "error" else "wrong")
                print(f"  [{architecture}] {task.identifier}: {episode.answer!r} ({verdict}; {episode.calls} calls)")

    summary = summarize(records, seeds)
    print(f"\n{'architecture':18s} {'accuracy':>15s} {'calls':>6s} {'tools':>6s} {'tokens':>8s} {'seconds':>8s} {'errors':>6s}")
    for name, row in sorted(summary.items(), key=lambda item: -item[1]["accuracy"]):
        print(
            f"{name:18s} {row['accuracy']:8.3f} ± {row['accuracy_std']:.3f} {row['calls_per_task']:6.1f} "
            f"{row['tool_calls_per_task']:6.1f} {row['tokens_per_task']:8.0f} {row['seconds_per_task']:8.1f} {row['errors']:6d}"
        )
    print(f"served by: {dict(served.most_common(5))}")

    with (Path(run_directory) / "transcripts.jsonl").open("w", encoding="utf-8") as handle:
        for transcript in transcripts:
            handle.write(json.dumps(transcript, ensure_ascii=False, default=str) + "\n")
    write_metrics(
        run_directory,
        records,
        summary=summary,
        served_by=dict(served),
        router=router.status() if router else [],
    )
    print(f"artifacts: {run_directory}")


if __name__ == "__main__":
    main()
