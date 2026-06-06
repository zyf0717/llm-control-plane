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
  min_confidence: 0.35

search:
  enabled: true
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
| `top_k` | Max retrieved chunks requested and injected |
| `min_confidence` | Minimum normalized confidence required for injection |

### Confidence Semantics

- If the RAG backend returns `score`, the proxy uses it directly.
- If the backend returns `distance`, the proxy converts confidence as `1.0 - distance`.

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
| `default_provider` | Default provider id when `provider=auto` |
| `timeout_ms` | Per-provider request timeout |
| `max_results` | Hard cap on normalized results returned |
| `cache_ttl_seconds` | Parsed-response cache TTL |
| `providers[].enabled` | Explicit allowlist toggle per provider |
| `providers[].priority` | Lower values are tried first |
| `providers[].fallback_only` | Only try after non-fallback providers |
| `providers[].cache_ttl_seconds` | Optional provider-specific TTL override |
| `providers[].min_interval_seconds` | Optional minimum delay between requests to the same provider |

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

## Operational Notes

- Smart routing relies on the current `reachable_endpoints` cache, which is refreshed by `/models`.
- Dashboard RAG choices are filtered by live `health_url` checks on page load and explicit RAG refresh.
- `default_endpoint` is only used if it is healthy and present in the configured list.

TL;DR: `config.yaml` controls upstream models, smart routing preference order, RAG discovery/retrieval behavior, and the optional lightweight search module.
