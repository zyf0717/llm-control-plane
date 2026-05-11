# Configuration

All runtime configuration lives in `config.yaml`.

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
  top_k: 5
  min_confidence: 0.35

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

## Operational Notes

- Smart routing relies on the current `reachable_endpoints` cache, which is refreshed by `/models`.
- Dashboard RAG choices are filtered by live `health_url` checks on page load and explicit RAG refresh.
- `default_endpoint` is only used if it is healthy and present in the configured list.

TL;DR: `config.yaml` controls upstream models, smart routing preference order, and RAG discovery/retrieval behavior.
