# Configuration

Routing and RAG configuration live in `config.yaml`; runtime environment overrides such as `HISTORY_DB_PATH` live in `.env` or the shell.

## Routing Example

```yaml
endpoints:
  evo-x2:
    url: https://llm-evo-x2.paperclips.dev
    model: halogen-qwen3.8-flash-next
    slot_affinity: true
  gti13:
    url: https://llm-gti13.paperclips.dev
    model: Qwen/Qwen3.5-4B

workloads:
  utility: [gti13, evo-x2]
  complex: [evo-x2, gti13]
  responsive: [gti13, evo-x2]
  vision: [evo-x2]
```

The complete template is [`config.example.yaml`](../config.example.yaml). The old
list-based endpoint/workload schema is no longer supported.

## LLM Endpoints and Models

Each endpoint key selects one URL and optional model. `url` is the upstream root,
without `/v1/chat/completions`. Optional `gpu`, `vram`, `cpu`, `ram`, and `soc`
metadata appears in discovery and response details. `slot_affinity` enables the
existing best-effort slot probe; it does not identify the serving software.

Devices may expose multiple URLs, and multiple endpoint keys may share a URL with
different explicit model IDs. For example, two entries using the same URL with
`model: small-model` and `model: large-model` select those models independently.

Model selection follows these rules:

- Use the configured model when it is advertised by `/v1/models`.
- An omitted or stale configured name uses the sole advertised model automatically.
- An empty catalogue is unavailable. Multiple advertised models require an explicit
  matching name; the proxy never chooses the first model. Configure one endpoint
  key per desired model, or expose each model on its own URL/port.
- If discovery fails, try the configured name. A definite model-not-found response
  refreshes discovery and permits one retry with the resolved model.
- An explicitly supplied request `model` is a strict constraint, rather than a
  recoverable configured default.

Discovery has a two-second probe limit and a shared 30-second cache per URL.
`GET /models` returns each endpoint's resolved model plus an `errors` mapping for
unresolved endpoints or failed discovery. Routing uses this same resolver and does
not depend on dashboard refreshes.

## Orchestration Subsystems

`orchestration.workflows.enabled` controls the YAML workflow subsystem. `orchestration.graphs.enabled` controls the LangGraph graph subsystem. Both default to enabled when omitted.

Each subsystem owns its own API, store, dashboard tab, and execution model. Disabling one should not affect the other.

## Repo-Context Configuration

`repo_context` configures the read-only local repository explorer used by `repo_context` workflow steps and the dashboard repository selector. When omitted, it defaults to enabled with `/home/yifei/repos` as the repository root and `/home/yifei/repos/repo-context` as the CLI project.

| Field | Meaning |
|---|---|
| `enabled` | Enable repo-context API and workflow execution |
| `project_dir` | Path passed to the default `uv run --project` entry point when `entry_point` is omitted |
| `repos_root` | Directory whose immediate child directories are valid repo targets |
| `entry_point.command` | CLI executable; `explore` is appended before request options |
| `entry_point.args` | CLI entry-point args before the `explore` subcommand |
| `entry_point.env` | Extra environment variables merged into the repo-context subprocess |
| `env` | Top-level subprocess environment overrides; applied after `entry_point.env` |
| `default_max_turns` | Step default when `repo_context_max_turns` is omitted |
| `timeout_seconds` | Subprocess timeout per repo-context step |
| `max_concurrent` | Max concurrent repo-context subprocesses |

## RAG Configuration

The `retrieval` block configures dashboard-visible RAG endpoints and proxy retrieval behavior.

### Fields

| Field | Meaning |
|---|---|
| `default_endpoint` | Retrieve URL selected by default if healthy |
| `endpoints[].name` | Display label in the dashboard |
| `endpoints[].retrieve_url` | POST endpoint used by the proxy for retrieval |
| `endpoints[].health_url` | GET endpoint used by the dashboard for health checks |
| `top_k` | Max retrieved chunks requested from the RAG service |

### Grounding Contract

The proxy sends the latest real user turn and `top_k` as `{query, limit}` to the normalized `/api/retrieve/context` route. The RAG service owns retrieval, ranking, thresholding, and prompt assembly. To inject context, it must return a non-empty `grounded_user_message`; the proxy rewrites only the latest user turn with that grounded message and does not persist the rewritten content to conversation history.

## Workload Preferences

Each workload maps directly to an ordered list of endpoint keys:

| Type | Usage |
|---|---|
| `utility` | Classifier, workflow/graph planners, query refinement, and LLM reranking |
| `complex` | Reasoning, analysis, and programming |
| `responsive` | General chat and simple requests |
| `vision` | Chat messages containing images |

`/smart` detects image content directly; other requests are classified as complex
or responsive through utility. The classifier disables reasoning and falls back
to responsive on failure. Explicit workload calls bypass classification.

Routing attempts only the configured preferences, in order. Connection failures,
HTTP 429, upstream 5xx, and unresolved model defaults permit the next candidate.
Other request errors propagate. A successful upstream response ends failover;
started streams are never replayed. Explicit endpoint requests and valid
conversation pins stay on their endpoint unless route switching is allowed.

Built-in search, RAG, and repo-context planners use utility in workflows and graphs;
final answers use the run's selected endpoint. Query refinement and LLM reranking
also use utility, with no separate URLs or placeholder model names. The dedicated
reranker keeps its own `/rerank` endpoint.

## Search Configuration

The `search` block enables the lightweight candidate-discovery module exposed at `POST /search/web`.

### Fields

| Field | Meaning |
|---|---|
| `enabled` | Enables the `/search/web` endpoint and internal provider router |
| `query_refiner_enabled` | Enables utility-backed query refinement; defaults to false |
| `query_refiner_timeout_ms` | Query-refiner request timeout; defaults to `timeout_ms` or 7000 |
| `query_refiner_max_source_chars` | Max optional context characters sent to the query refiner |
| `query_refiner_max_output_tokens` | Max query-refiner response tokens |
| `query_refiner_max_queries` | Max refined query fanout; defaults to 1 |
| `reranker_endpoint` | Root URL for the dedicated `/rerank` service |
| `reranker_backend` | Reranker protocol: `llm` for OpenAI chat completions, `dedicated` for `/rerank`; defaults to `llm` |
| `reranker_model` | Optional display label for the dedicated reranker; LLM metadata uses the actual served model |
| `reranker_enabled` | Enables reranking; defaults to true when `reranker_endpoint` is configured |
| `reranker_timeout_ms` | Reranker request timeout; defaults to `timeout_ms` or 7000 |
| `reranker_max_source_chars` | Max optional context characters sent to the reranker |
| `reranker_max_candidates` | Max deduped candidates sent to the reranker |
| `reranker_max_output_tokens` | Max reranker response tokens |
| `search_max_total_results` | Max deduped results returned across all refined queries |
| `default_provider` | Default provider id when `provider=auto` |
| `timeout_ms` | Per-provider request timeout |
| `max_results` | Hard cap on normalized results returned |
| `cache_ttl_seconds` | Parsed-response cache TTL |
| `providers[].enabled` | Explicit allowlist toggle per provider |
| `providers[].priority` | Lower values are tried first |
| `providers[].fallback_only` | Only try after non-fallback providers |
| `providers[].cache_ttl_seconds` | Optional provider-specific TTL override |
| `providers[].min_interval_seconds` | Optional minimum delay between requests to the same provider |

When query refinement is enabled, ad hoc `/search/web` calls may run a bounded, non-persisted query-refiner call before provider execution. The query refiner returns one or more concise provider-ready queries; the first query becomes the effective search query. If `query_refiner_max_queries` is greater than 1, provider searches run as an async fanout and results are deduped before applying `search_max_total_results`. Query-refiner failure degrades to the original query and adds a warning.

When reranking is enabled, `reranker_backend: llm` calls utility, while
`reranker_backend: dedicated` posts `{query, documents, top_k}` to the configured
`reranker_endpoint` plus `/rerank`. A dedicated failure falls back to utility.
If both fail, provider order is preserved with a warning. Metadata reports the
actual served LLM model and the path used: `dedicated`, `llm`, or `none`.

Workflow search can use either query-refiner planning or workflow-owned planning. Plain search prompts may use the query refiner. Workflow-planned JSON queries should set `use_query_refiner: false`. Search steps may set `search_count` to control provider candidates per dispatched query. Workflow search steps never rerank inline; `use_reranker` and `rerank_context` on search steps are invalid. Post-retrieval reranking must be modeled as an explicit `rerank` step that depends on the retrieval step. Rerank steps may set `rerank_top_k` to cap the returned reranked results.

LLM workflow steps can declare machine-readable output boundaries with `output_contract`. Supported formats are `json`, `yaml`, and `text`; `json` and `yaml` contracts parse the whole model response, validate it with JSON Schema, preserve raw text in `output_json.text`, and persist the parsed value in `output_json.json`. New contracts default to bounded correction when `on_invalid` is omitted: one repair attempt followed by one retry. The former `output_schema` shorthand is not supported; use `output_contract.schema`.

```yaml
output_contract:
  format: json
  required: true
  schema:
    type: object
    additionalProperties: false
    required: [queries]
    properties:
      queries:
        type: array
        minItems: 1
        items:
          type: string
  on_invalid:
    action: retry
    max_attempts: 2
    repair: true
```

Supported built-in provider ids:

- `duckduckgo_html`
- `marginalia_html`
- `mojeek_html`
- `wikipedia_opensearch`
- `searxng_html` when `providers.searxng_html.endpoint` is configured


## Runtime Environment

Supported environment variables:

- `API_KEY_ID`
- `API_KEY_SECRET`
- `PROXY_BASE_URL`
- `HISTORY_DB_PATH` to override the default local SQLite history path (`var/history.sqlite3`)

Blank `HISTORY_DB_PATH` is treated the same as unset and uses the default SQLite path. In-memory history is available only when explicitly injected by tests or alternate runtime wiring.

## Operational Notes

- Smart routing resolves models lazily through the shared discovery cache.
- The dashboard uses concrete endpoint selections and does not keep a separate local route pin.
- Dashboard Retrieval choices show health status on page load and explicit refresh.
- Retrieval selection is explicit; the dashboard retains configured choices even when unhealthy.

TL;DR: `config.yaml` controls upstream models, smart routing preference order, RAG discovery/retrieval behavior, and the optional lightweight search module.
