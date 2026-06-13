# Configuration

Routing and RAG configuration live in `config.yaml`; runtime environment overrides such as `HISTORY_DB_PATH` live in `.env` or the shell.

## Full Example

```yaml
endpoints:
  - name: "my-server"
    gpu: "NVIDIA GeForce RTX 4090"
    vram: "24GB"
    cpu: "Intel Core i9-13900K"
    ram: "64GB"
    url: "https://my-llm-server.example.com"

  - name: "mac-mini"
    soc: "Apple M4"
    ram: "16GB"
    url: "https://mac-mini.example.com"

rag:
  default_endpoint: "http://localhost:8100/api/retrieve"
  endpoints:
    - name: "localhost:8100"
      retrieve_url: "http://localhost:8100/api/retrieve"
      health_url: "http://localhost:8100/api/health"
  top_k: 10

search:
  enabled: true
  query_refiner_model_endpoint: "https://llm-server.example.com"
  query_refiner_model: "search-query-refiner"
  query_refiner_enabled: true
  query_refiner_timeout_ms: 7000
  query_refiner_max_context_chars: 24000
  query_refiner_max_output_tokens: 1024
  query_refiner_max_queries: 3
  reranker_model_endpoint: "http://reranker-server.example.com"
  reranker_fallback_model_endpoint: "https://llm-server.example.com"
  reranker_backend: "dedicated"
  reranker_model: "search-reranker"
  reranker_enabled: true
  reranker_timeout_ms: 7000
  reranker_max_context_chars: 12000
  reranker_max_candidates: 20
  reranker_max_output_tokens: 1024
  search_max_total_results: 10
  default_provider: "duckduckgo_html"
  timeout_ms: 7000
  max_results: 10
  cache_ttl_seconds: 900
  providers:
    duckduckgo_html:
      enabled: true
      priority: 10
    marginalia_html:
      enabled: true
      priority: 20
    mojeek_html:
      enabled: false
      priority: 30

workloads:
  - type: "reasoning"
    endpoint_preference: ["my-server", "mac-mini"]
  - type: "programming"
    endpoint_preference: ["my-server", "mac-mini"]
  - type: "tokens_per_second"
    endpoint_preference: ["my-server", "mac-mini"]
  - type: "ttft_content"
    endpoint_preference: ["mac-mini", "my-server"]
  - type: "classification"
    endpoint_preference: ["mac-mini", "my-server"]
```

## LLM Endpoints

Each item in `endpoints` defines one upstream model host.

Required fields:

- `name`
- `url`

Optional metadata fields:

- `gpu`
- `vram`
- `cpu`
- `ram`
- `soc`

The metadata is surfaced through `/models` and shown in the dashboard runtime panel.

## RAG Configuration

The `rag` block configures dashboard-visible RAG endpoints and proxy retrieval behavior.

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

Supported workload types:

| Type | Usage |
|---|---|
| `reasoning` | Multi-step analysis and trade-off tasks |
| `programming` | Coding, debugging, or code explanation |
| `tokens_per_second` | Long output with minimal deep reasoning |
| `ttft_content` | Fast first-token, simple Q&A, general chat |
| `classification` | Internal workload classification endpoint choice |

Each `endpoint_preference` list is ordered from most preferred to least preferred.

## Search Configuration

The `search` block enables the lightweight candidate-discovery module exposed at `POST /search/web`.

### Fields

| Field | Meaning |
|---|---|
| `enabled` | Enables the `/search/web` endpoint and internal provider router |
| `query_refiner_model_endpoint` | Optional OpenAI-compatible endpoint for bounded query refinement |
| `query_refiner_model` | Query-refiner model name; defaults to `search-query-refiner` |
| `query_refiner_enabled` | Enables or disables query refinement; defaults to true when `query_refiner_model_endpoint` exists |
| `query_refiner_timeout_ms` | Query-refiner request timeout; defaults to `timeout_ms` or 7000 |
| `query_refiner_max_context_chars` | Max optional context characters sent to the query refiner |
| `query_refiner_max_output_tokens` | Max query-refiner response tokens |
| `query_refiner_max_queries` | Max refined query fanout; defaults to 1 |
| `reranker_model_endpoint` | Optional reranker endpoint for bounded post-retrieval reranking |
| `reranker_fallback_model_endpoint` | Optional OpenAI-compatible LLM reranker endpoint used if a dedicated reranker fails |
| `reranker_backend` | Reranker protocol: `llm` for OpenAI chat completions, `dedicated` for `/rerank`; defaults to `llm` |
| `reranker_model` | Reranker model name; defaults to `search-reranker` |
| `reranker_enabled` | Enables or disables reranking; defaults to true when `reranker_model_endpoint` exists |
| `reranker_timeout_ms` | Reranker request timeout; defaults to `timeout_ms` or 7000 |
| `reranker_max_context_chars` | Max optional context characters sent to the reranker |
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

When `search.query_refiner_model_endpoint` is configured and query refinement is enabled, ad hoc `/search/web` calls may run a bounded, non-persisted query-refiner call before provider execution. The query refiner returns one or more concise provider-ready queries; the first query becomes the effective search query. If `query_refiner_max_queries` is greater than 1, provider searches run as an async fanout and results are deduped before applying `search_max_total_results`. Query-refiner failure degrades to the original query and adds a warning.

When `search.reranker_model_endpoint` is configured and reranking is enabled, search runs a bounded, non-persisted reranker call after provider retrieval and dedupe. `reranker_backend: llm` uses `/v1/chat/completions`; `reranker_backend: dedicated` posts `{query, documents, top_k}` to `/rerank`. A dedicated reranker can fall back to `reranker_fallback_model_endpoint`; otherwise reranker failure preserves provider order and adds a warning. Reranking metadata exposes the configured `backend` and the actual `path` used for the returned ranking: `dedicated`, `llm`, or `none`.

Workflow search can use either query-refiner planning or workflow-owned planning. Plain search prompts may use the query refiner. Workflow-planned JSON queries should set `use_query_refiner: false`. Workflow search steps never rerank inline; `use_reranker` and `rerank_context` on search steps are invalid. Post-retrieval reranking must be modeled as an explicit `rerank` step that depends on the retrieval step.

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

- Smart routing relies on the current `reachable_endpoints` cache, which is refreshed by `/models`, and is exposed through the proxy `/smart` route.
- The dashboard uses concrete machine endpoint selections and does not keep a separate local route pin.
- Dashboard RAG choices are filtered by live `health_url` checks on page load and explicit RAG refresh.
- `default_endpoint` is only used if it is healthy and present in the configured list.

TL;DR: `config.yaml` controls upstream models, smart routing preference order, RAG discovery/retrieval behavior, and the optional lightweight search module.
