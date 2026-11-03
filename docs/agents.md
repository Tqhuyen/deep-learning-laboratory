# Agents on free models

`laboratory/agents/` runs LLM agent **architectures** (ReAct, native tool
calling, plan-and-execute, Reflexion, self-consistency, debate) on top of
**free** hosted chat models. It trains nothing and does not import torch: the
questions are about control flow — how to plan, act, reflect and vote — not
about weights.

Every model it calls costs $0: the 37 chat providers listed on
<https://freellmapi.co/models> and the `:free` models of
<https://unorouter.com/en/models?max_price=0>, plus two local options (a
self-hosted FreeLLMAPI router and any OpenAI-compatible server).

## 1. Pipeline and files

The area has its own small pipeline, one file per family:

```text
environment -> providers -> catalog -> wire / client -> routing -> tools
    -> architectures (loops, planning, reflection, ensembles) -> tasks
```

| file | what it holds |
|---|---|
| `environment.py` | `.env` loader (`load_environment`); shell variables win over the file |
| `providers.py` | `Provider` / `Endpoint`, the 37 `CATALOG_PROVIDERS` plus `LOCAL_PROVIDERS` (`freellmapi`, `custom`) |
| `catalog.py` | `data/free_models.json` snapshot, `family_key`, `select_endpoints` model specifications |
| `wire.py` | Chat Completions and Responses wire protocols, `ProviderError` kinds, rate-limit headers |
| `client.py` | `ChatClient`: one HTTP call over `urllib`, replaceable transport, model-identity check |
| `routing.py` | `ModelRouter` (failover, pacing, cool-downs), `connect()`, `live_rosters` |
| `messages.py` | `Message`, `ToolCall`, `Completion`, `Usage`, the `ChatModel` protocol |
| `tools.py` | `Tool`, `Toolbox`, `calculator`, `lookup_tool` |
| `scripted.py` | `ScriptedModel` and `tool_call_message`: the offline model for `tiny` and unit tests |
| `loops.py` | `Episode`, `Agent`, `DirectAgent`, `ReActAgent`, `ToolCallingAgent` |
| `planning.py` | `PlanAndExecuteAgent` |
| `reflection.py` | `ReflexionAgent` |
| `ensembles.py` | `SelfConsistencyAgent`, `DebateAgent` |
| `tasks.py` | `TaskSuite`, `load_task_suite`, `answers_match` over `data/agent_tasks.json` |

Data and scripts:

| path | content |
|---|---|
| `data/free_models.json` | 523 free chat models of 37 providers, 298 with tool support |
| `data/prepare_free_models.py` | refreshes the snapshot from the two public catalogs |
| `data/agent_tasks.json` | 24 tasks over a fictional encyclopedia |
| `.env.example` | every provider variable with its sign-up URL |
| `experiments/agents/provider_survey/run.py` | which free models answer for your keys |
| `experiments/agents/architecture_comparison/run.py` | architectures on the same tasks and model |

## 2. Setup

```bash
cp .env.example .env      # then fill in the keys you have; .env is git-ignored
python experiments/agents/provider_survey/run.py --tier smoke
```

Four providers answer with **no key at all**: `kilo`, `ovh`, `llm7` and
`aihorde`. A key only raises their limits. With an empty `.env` the survey and
the comparison still run on those four.

Environment variables follow one rule, so nothing has to be looked up:

| variable | meaning |
|---|---|
| `<ID>_API_KEY` | the key, e.g. `GROQ_API_KEY`, `UNOROUTER_API_KEY` |
| `<ID>_BASE_URL` | optional base-URL override, e.g. `ZHIPU_BASE_URL` |
| `CLOUDFLARE_ACCOUNT_ID` | Cloudflare only: part of the URL, needed next to `CLOUDFLARE_API_KEY` |

Special cases:

* **Zhipu.** The default URL is Z.ai (`https://api.z.ai/api/paas/v4`). Keys
  from open.bigmodel.cn need
  `ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4`.
* **FreeLLMAPI self-hosted router**
  (<https://github.com/tashfeenahmed/freellmapi>). Set `FREELLMAPI_BASE_URL`
  (default `http://localhost:3001/v1`) and `FREELLMAPI_API_KEY`, then use
  `--model freellmapi:auto` (also `auto:fast`, `auto:smart`). The router does its
  own failover across the providers above.
* **Local OpenAI-compatible server** (Ollama `http://localhost:11434/v1`, LM
  Studio `http://localhost:1234/v1`, vLLM, llama.cpp). Set `CUSTOM_BASE_URL`; a
  key is optional. Use `--model custom:<model>`, e.g. `custom:llama3.2`.

## 3. Providers

Generated from `CATALOG_PROVIDERS` and `data/free_models.json` (snapshot in
the repository). "with tools" counts catalog rows that list tool support.

| id | name | key variable | keyless | free models | with tools | notes |
|---|---|---|---|---:|---:|---|
| `kilo` | Kilo Gateway | `KILO_API_KEY` | yes | 9 | 7 | :free routes answer anonymously, 200 requests/hour per IP; free prompts may be logged |
| `ovh` | OVH AI Endpoints | `OVH_API_KEY` | yes | 12 | 9 | anonymous tier: 2 requests/min per IP per model |
| `llm7` | LLM7 | `LLM7_API_KEY` | yes | 3 | 2 | anonymous access works; a free token raises the limits |
| `aihorde` | AI Horde | `AIHORDE_API_KEY` | yes | 32 | 0 | volunteer GPUs: slow queue, rotating roster, no tool calling; a key raises queue priority; live roster (GET /models) |
| `google` | Google AI Studio | `GOOGLE_API_KEY` |  | 15 | 12 | per-model RPM/RPD caps; hidden thinking tokens count against max_tokens |
| `groq` | Groq | `GROQ_API_KEY` |  | 7 | 4 | free key, no card; per-model RPM/RPD caps |
| `mistral` | Mistral | `MISTRAL_API_KEY` |  | 12 | 11 | free Experiment plan |
| `cohere` | Cohere | `COHERE_API_KEY` |  | 17 | 9 | trial key with monthly call caps |
| `nvidia` | NVIDIA NIM | `NVIDIA_API_KEY` |  | 18 | 8 | about 40 RPM per account; evaluation-only terms; one tool call per turn |
| `cloudflare` | Cloudflare Workers AI | `CLOUDFLARE_API_KEY` |  | 26 | 12 | 10,000 neurons/day; API token with Workers AI permission plus CLOUDFLARE_ACCOUNT_ID |
| `zhipu` | Zhipu AI (Z.ai) | `ZHIPU_API_KEY` |  | 3 | 3 | keys from open.bigmodel.cn need ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4 |
| `modelscope` | ModelScope | `MODELSCOPE_API_KEY` |  | 11 | 0 | the token works only after binding a real-name Alibaba Cloud China account |
| `sealion` | SEA-LION | `SEALION_API_KEY` |  | 6 | 1 | AI Singapore; 10 RPM |
| `agnes` | Agnes AI | `AGNES_API_KEY` |  | 5 | 2 | promotional $0 pricing on Agnes models |
| `radeon` | AMD Radeon Cloud | `RADEON_API_KEY` |  | 2 | 2 | rotating free roster; one tool call per turn; long reasoning |
| `moondream` | Moondream | `MOONDREAM_API_KEY` |  | 2 | 0 | vision model; $5/month shared credits; no tools or stop sequences; model identity verified |
| `unorouter` | UnoRouter | `UNOROUTER_API_KEY` |  | 109 | 81 | :free models: 1 request/min per model; bursts trip an account-wide 429; shared limit |
| `openrouter` | OpenRouter | `OPENROUTER_API_KEY` |  | 9 | 5 | :free models share 50 requests/day per account (1,000 after a one-time $10 top-up); shared limit |
| `xkiro` | xKiro | `XKIRO_API_KEY` |  | 28 | 28 | 5M free tokens/day; premium models answer 403 |
| `electronhub` | ElectronHub | `ELECTRONHUB_API_KEY` |  | 60 | 55 | shared weekly free credits |
| `requesty` | Requesty | `REQUESTY_API_KEY` |  | 8 | 5 | free key, no card |
| `routeway` | Routeway | `ROUTEWAY_API_KEY` |  | 7 | 1 | :free models, roughly 5-20 RPM |
| `nara` | NaraRouter | `NARA_API_KEY` |  | 9 | 8 | daily free quota after Telegram verification |
| `bazaarlink` | BazaarLink | `BAZAARLINK_API_KEY` |  | 3 | 3 | only the auto:free route is free |
| `aion` | Aion Labs | `AION_API_KEY` |  | 4 | 0 | free key, no card |
| `router9` | Router9 | `ROUTER9_API_KEY` |  | 5 | 0 | shared monthly free credits |
| `speka` | Speka | `SPEKA_API_KEY` |  | 5 | 0 | shared $1/month free credits |
| `septor` | Septor Labs | `SEPTOR_API_KEY` |  | 5 | 4 | shared daily free quota; several aliases substitute another model; model identity verified |
| `clod` | CLōD | `CLOD_API_KEY` |  | 4 | 0 | shared daily free requests; model identity verified |
| `airforce` | Api.Airforce | `AIRFORCE_API_KEY` |  | 20 | 0 | 1 request/min per account; shared limit; model identity verified |
| `dreamprompting` | DreamPrompting | `DREAMPROMPTING_API_KEY` |  | 14 | 5 | 500K tokens and 5,000 requests per rolling 24 h; pooled router; model identity verified |
| `waterfall` | Waterfall | `WATERFALL_API_KEY` |  | 10 | 0 | community free models; model identity verified |
| `logfare` | Logfare | `LOGFARE_API_KEY` |  | 2 | 0 | fair-use free tier that logs all traffic; model identity verified |
| `llmtr` | LLMTR | `LLMTR_API_KEY` |  | 2 | 0 | daily free-model quotas; automated key probing gets accounts suspended; model identity verified |
| `gizmo` | Gizmo | `GIZMO_API_KEY` |  | 6 | 0 | monthly free requests; text only, no tools |
| `blockrun` | BlockRun | `BLOCKRUN_API_KEY` |  | 4 | 0 | zero-priced routes; model identity verified |
| `aclide` | ACLIDE | `ACLIDE_API_KEY` |  | 29 | 21 | OpenAI Responses API; shared monthly credits; closed to new sign-ups; wire=responses |

"shared limit" is `Provider.shared_limit`; "model identity verified" is
`Provider.verify_model`; ACLIDE is the only provider on `wire="responses"`.
AI Horde's catalog rows are ignored at run time: `connect()` asks its
`GET /models` for the workers online now.

## 4. Catalog

`data/free_models.json` is a committed snapshot of two public catalogs:

| source | URL | contributes |
|---|---|---|
| FreeLLMAPI | `https://api.freellmapi.co/v1/latest` | every free chat model of 37 providers, RPM/RPD, context, tool support, intelligence rank |
| UnoRouter | `https://unorouter.com/api/models/pricing/browse` | every `:free` text model; replaces FreeLLMAPI's partial UnoRouter rows |

```bash
python data/prepare_free_models.py      # network; rewrites data/free_models.json
```

Free tiers change weekly. A stale row costs one failed request, after which the
router drops that endpoint for the run.

* **Families.** `family_key` normalises a display name the way freellmapi.co
  does (`deepseek-v4-flash`), so one family can be served by several providers
  and failover changes the provider, never the weights.
* **Guard models excluded.** Safety classifiers (FreeLLMAPI quirk
  `safety-classifier-output`, UnoRouter names matching `guard` or
  `content-safety`) stay in the snapshot with `guard=true` but are never
  selected: they label text instead of answering it.
* **Only chat.** Image, audio, TTS and embedding rows are dropped by the
  parsers. Speechify, listed on freellmapi.co, is TTS-only and therefore not a
  provider here: an agent needs a model that returns chat text.

## 5. Model specifications

`connect(specification)` and `--model` take a comma-separated chain; each part
is one of:

| part | example | selects |
|---|---|---|
| `auto` | `auto` | every configured free model, strongest first |
| family | `gpt-oss-120b` | that model on every configured provider |
| provider | `groq` | every free model of that provider |
| `provider:model` | `ovh:gpt-oss-120b`, `custom:llama3.2`, `freellmapi:auto` | one endpoint, which may be outside the catalog |

* Parts are concatenated in order, duplicates removed: `groq,auto` tries Groq
  first, then everything else.
* Within a part, rows are ordered by catalog rank (FreeLLMAPI
  `intelligenceRank`, unknown last), then by RPM, highest first.
* `require_tools=True` keeps rows whose catalog lists tool support on providers
  that accept `tools`. The comparison sets it whenever a selected architecture
  needs native tools.
* Unconfigured providers are skipped silently for `auto`, family and provider
  parts; an explicit `provider:model` that is not configured raises and names
  the missing variables.
* AI Horde is a live roster: `connect()` queries its `GET /models` (unless the
  whole chain is explicit `provider:model` parts or `discover=False`).

## 6. Routing

`ModelRouter` is a `ChatModel` over endpoints in priority order. Per request it
calls the first ready endpoint; on failure it reacts to `ProviderError.kind`:

| kind | typical cause | router action |
|---|---|---|
| `rate_limited` | 429, UnoRouter 503 `get_channel_failed` | cool down for Retry-After, or 30 s doubling per consecutive strike (max 1 h) |
| `transient` | 5xx, timeout, malformed or in-band error | short cool-down (20 s), fail over |
| `unauthorized` | 401, "API key not valid" | drop every endpoint of the provider |
| `unavailable` | 402/403/404, unknown or premium model, substituted model | drop the endpoint for the rest of the run |
| `bad_request` | 400/413/422, empty completion, request this model rejects | skip it for this request only |

* **RPM pacing.** Each endpoint is paced to its catalog RPM over a sliding
  60 s window.
* **Shared-limit providers** (`unorouter`, `openrouter`, `airforce`): a 429 on
  one model cools down every endpoint of the account.
* **Exhausted windows.** A successful response whose rate-limit headers report
  zero remaining requests cools the endpoint down until the window resets.
* **`max_wait`.** When every candidate is cooling down the router sleeps until
  the first is ready, up to `max_wait` seconds (default 180), so long runs ride
  through per-minute caps.
* **`RouterExhausted`.** Raised when no endpoint can serve, all are cooling
  longer than `max_wait`, or `max_attempts` (default 8) is spent. Its `trail`
  lists every failed attempt as `provider:model: kind: message`.
* **`Completion.trail`.** A successful completion carries the failures that
  preceded it, so a transcript shows the path a request took.
* **Model identity.** `Completion.served_model` is the model the response
  reported (or the `x-routed-via` header). Gateways that substitute models
  silently have `verify_model=True`; a mismatch there becomes `unavailable`
  (except on `auto` routes, where substitution is the point).

## 7. Python usage

ReAct on the task suite, over every configured free model:

```python
from laboratory.agents import ReActAgent, answers_match, connect, load_task_suite

suite = load_task_suite()
model = connect("auto", max_wait=120)          # or "gpt-oss-120b", "groq", "ovh:gpt-oss-120b"
agent = ReActAgent(model, suite.toolbox(), max_steps=6)
for task in suite.tasks[:4]:
    episode = agent.run(task.question)
    print(task.identifier, episode.answer, answers_match(episode.answer, task.answer),
          episode.calls, episode.completions[-1].endpoint if episode.completions else None)
```

One call to one endpoint, no router:

```python
import os
from laboratory.agents import ChatClient, Message, load_environment, resolve_endpoint

load_environment()
endpoint = resolve_endpoint("ovh", "gpt-oss-120b", os.environ)   # None when not configured
completion = ChatClient().complete(endpoint, [Message.user("Reply with exactly: pong")])
print(completion.text, completion.served_model, completion.usage)
```

Offline, deterministic, for tests:

```python
from laboratory.agents import ScriptedModel, ToolCallingAgent, calculator, Toolbox, tool_call_message

model = ScriptedModel([tool_call_message("calculator", {"expression": "6 * 7"}), "Final Answer: 42"])
episode = ToolCallingAgent(model, Toolbox([calculator()])).run("What is 6 * 7?")
assert episode.answer == "42" and len(model.requests) == 2
```

## 8. Architectures

| name | class | file | paper | calls per task (rough) | native tools? |
|---|---|---|---|---|---|
| `direct` | `DirectAgent` | `loops.py` | — (the floor) | 1 | no (no tools) |
| `react` | `ReActAgent` | `loops.py` | [Yao 2022](https://arxiv.org/abs/2210.03629) | 2 – `max_steps`+1 | no: text Thought/Action/Observation |
| `tool_calling` | `ToolCallingAgent` | `loops.py` | OpenAI function calling | 2 – `max_steps`+1 | yes |
| `plan_and_execute` | `PlanAndExecuteAgent` | `planning.py` | [Wang 2023](https://arxiv.org/abs/2305.04091) | 1 plan + one inner run per step + 1 solve | as the inner agent |
| `reflexion` | `ReflexionAgent` | `reflection.py` | [Shinn 2023](https://arxiv.org/abs/2303.11366) | per trial: inner run + evaluation + reflection | as the inner agent |
| `self_consistency` | `SelfConsistencyAgent` | `ensembles.py` | [Wang 2022](https://arxiv.org/abs/2203.11171) | `samples` × inner run | as the inner agent |
| `debate` | `DebateAgent` | `ensembles.py` | [Du 2023](https://arxiv.org/abs/2305.14325) | `debaters` × `rounds` inner runs | as the inner agent |

Composites wrap an inner agent and inherit its `requires_tools`.
`--inner react` (default) runs them on any chat model; `--inner tool_calling`
needs tool-capable endpoints. Self-consistency and debate sample their inner
agent at `sample_temperature` (0.7) for diversity. Reflexion's default
evaluator is the model itself, which never sees the reference answer; pass
`evaluator=` for a real signal.

Every architecture returns an `Episode`: answer, `stopped`
(`answer`, `max_steps` or `error`), transcript, every `Completion` and
architecture-specific `events`. A model failure ends the episode with
`stopped="error"` instead of raising.

## 9. Experiments

| tier | provider survey | architecture comparison |
|---|---|---|
| `tiny` | 2 scripted offline endpoints; no network | 1 task, all 7 architectures on `ScriptedModel`; no network |
| `smoke` | 1 model per configured provider | 4 tasks, `direct`, `react`, `tool_calling` |
| `development` | 3 models per provider | 12 tasks, all 7 architectures |
| `full` | every free model | 24 tasks × 3 seeds, all 7 architectures |

```bash
python experiments/agents/provider_survey/run.py --tier tiny
python experiments/agents/provider_survey/run.py --tier smoke
python experiments/agents/provider_survey/run.py --tier development --providers groq,ovh,kilo
python experiments/agents/provider_survey/run.py --tier full
python experiments/agents/provider_survey/run.py --tier smoke --models custom:llama3.2,freellmapi:auto

python experiments/agents/architecture_comparison/run.py --tier tiny
python experiments/agents/architecture_comparison/run.py --tier smoke
python experiments/agents/architecture_comparison/run.py --tier development --model gpt-oss-120b
python experiments/agents/architecture_comparison/run.py --tier full --model gpt-oss-120b
python experiments/agents/architecture_comparison/run.py --tier development --architectures react,reflexion --inner tool_calling
```

`--steps` overrides `models_per_provider` (survey) or `max_steps` (comparison).
The survey sends an echo probe ("Reply with exactly: pong") to every endpoint,
then a calculator tool-call probe (`1234 * 5678`) to those listed with tools.

Artifacts in `runs/agents/<name>/<tier>/` (no checkpoint):

| file | provider survey | architecture comparison |
|---|---|---|
| `config.json` | resolved preset | preset, architectures, seeds, `model`, `endpoints`, task ids |
| `metrics.json` | one row per endpoint (`answered`, `followed`, `served_model`, `latency`, `tools_listed`, `tool_ok`, `error`); `providers` summary (`endpoints`, `answered`, `followed`, `tool_ok`, `median_latency`); `answered`; `tool_ready` | one record per seed × architecture × task; `summary` per architecture (`accuracy`, `accuracy_std`, `accuracy_by_seed`, `accuracy_by_kind`, `calls_per_task`, `tool_calls_per_task`, `tokens_per_task`, `seconds_per_task`, `errors`, `out_of_steps`); `served_by`; `router` status |
| `transcripts.jsonl` | every probe request and reply | every episode: messages, events, completions |

## 10. Extending

* **Architecture.** Subclass `Agent` in the matching family file
  (`loops.py`, `planning.py`, `reflection.py`, `ensembles.py`, or a new file
  for a new family), implement `solve(self, episode)` using `self.call(...)`,
  set `requires_tools` if it needs native tools, and register a builder
  `name -> builder(model, toolbox, config)` in `BUILDERS` in
  `experiments/agents/architecture_comparison/run.py`. Add its offline script to
  `TINY_SCRIPTS` so the `tiny` tier covers it.
* **Tool.** `Tool(name, description, parameters, function)` where
  `parameters` is the JSON schema of the arguments object. The same object is
  sent as a native function definition and rendered as text for ReAct. Shared
  tools go in `tools.py`.
* **Provider.** Add a `_provider(...)` entry to `CATALOG_PROVIDERS` in
  `providers.py` (quirk fields for every divergence from Chat Completions), add
  its `<ID>_API_KEY=` line with the sign-up URL to `.env.example`, and refresh
  the catalog with `python data/prepare_free_models.py`.
* **Tasks.** `data/agent_tasks.json` has `knowledge` (title -> entry text, the
  lookup tool's encyclopedia) and `tasks`, each
  `{"id", "kind", "question", "answer"}` with `kind` one of `arithmetic`,
  `lookup`, `multi_hop`, `lookup_arithmetic`. Keep facts fictional so answers
  must come from the tool.

## 11. Fair comparisons and caveats

* **Pin a family** (`--model gpt-oss-120b`) when comparing architectures.
  `auto` mixes models of different strength across requests.
* **Free tiers change weekly.** Models disappear, limits move; re-run the
  survey and refresh the catalog before a serious run, and record the date.
* **Nondeterminism.** Remote models are not seeded; even temperature 0 varies
  across providers and days. Use the `full` tier's seeds and report
  `accuracy_std`.
* **Report `served_by`** from `metrics.json`: which endpoints actually
  answered, and check `served_model` in the transcripts.
* **Privacy.** Some free tiers log prompts or may train on them: Kilo (free
  prompts may be logged), Logfare (logs all traffic) and UnoRouter models marked
  "Provider may train on or retain prompts" in the catalog notes. Send nothing
  private.
* **Never commit `.env`.** It is git-ignored; keys never enter configs or run
  artifacts.
* This is for personal experimentation. Respect each provider's terms (NVIDIA:
  evaluation only; LLMTR suspends accounts that probe keys automatically).
