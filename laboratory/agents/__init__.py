"""LLM agents over free remote models.

This area is the one part of the laboratory that trains nothing: it runs agent
*architectures* on top of free hosted models, so the questions are about
control flow (how to plan, act, reflect, vote) rather than weights. It has its
own small pipeline, one file per family:

``environment`` (``.env`` keys) -> ``providers`` (37 free APIs) -> ``catalog``
(which models are free) -> ``wire`` / ``client`` (one HTTP call) -> ``routing``
(failover and rate limits) -> ``tools`` -> architectures (``loops``,
``planning``, ``reflection``, ``ensembles``) -> ``tasks`` (suites and scoring).

Quick start::

    from laboratory.agents import ReActAgent, connect, load_task_suite

    suite = load_task_suite()
    model = connect("auto")              # every configured free model, strongest first
    episode = ReActAgent(model, suite.toolbox()).run(suite.tasks[0].question)
    print(episode.answer, episode.calls, episode.completions[-1].endpoint)

Nothing here imports torch; :mod:`laboratory.agents.scripted` provides the
offline model the ``tiny`` tier and the unit tests run on.
"""

from .catalog import FreeModel, family_key, load_catalog, select_endpoints
from .client import ChatClient, HttpResponse
from .ensembles import DebateAgent, SelfConsistencyAgent, majority_vote
from .environment import load_environment, parse_environment_file
from .loops import (
    Agent,
    DirectAgent,
    Episode,
    ReActAgent,
    ToolCallingAgent,
    canonical_answer,
    extract_final_answer,
)
from .messages import ChatModel, Completion, Message, ToolCall, Usage
from .planning import PlanAndExecuteAgent
from .providers import (
    CATALOG_PROVIDERS,
    PROVIDERS,
    Endpoint,
    Provider,
    configured_providers,
    resolve_endpoint,
)
from .reflection import ReflexionAgent
from .routing import ModelRouter, RouterExhausted, connect
from .scripted import ScriptedModel, tool_call_message
from .tasks import Task, TaskSuite, answers_match, load_task_suite
from .tools import Tool, Toolbox, calculator, lookup_tool
from .wire import ProviderError

__all__ = [
    "CATALOG_PROVIDERS",
    "PROVIDERS",
    "Agent",
    "ChatClient",
    "ChatModel",
    "Completion",
    "DebateAgent",
    "DirectAgent",
    "Endpoint",
    "Episode",
    "FreeModel",
    "HttpResponse",
    "Message",
    "ModelRouter",
    "PlanAndExecuteAgent",
    "Provider",
    "ProviderError",
    "ReActAgent",
    "ReflexionAgent",
    "RouterExhausted",
    "ScriptedModel",
    "SelfConsistencyAgent",
    "Task",
    "TaskSuite",
    "Tool",
    "ToolCall",
    "ToolCallingAgent",
    "Toolbox",
    "Usage",
    "answers_match",
    "calculator",
    "canonical_answer",
    "configured_providers",
    "connect",
    "extract_final_answer",
    "family_key",
    "load_catalog",
    "load_environment",
    "load_task_suite",
    "lookup_tool",
    "majority_vote",
    "parse_environment_file",
    "resolve_endpoint",
    "select_endpoints",
    "tool_call_message",
]
