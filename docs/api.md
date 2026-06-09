# API Guide

The proxy listens on `http://localhost:12340` and accepts OpenAI-style chat-completions requests.

## Direct Endpoint Routing

Send directly to a configured endpoint:

```text
POST /{endpoint-name}
```

Example:

```bash
curl -X POST http://localhost:12340/my-server \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.3-70b",
    "messages": [{"role": "user", "content": "Explain transformers briefly."}]
  }'
```

## Smart Routing

Smart routing uses the latest user message to classify workload and select a reachable endpoint. Without `X-Convo-ID`, routing remains stateless. With `X-Convo-ID`, the proxy pins the first route decision server-side and reuses it on later `/smart` calls unless the configured endpoint is removed.

```text
POST /smart
POST /
```

Example:

```bash
curl -X POST http://localhost:12340/smart \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a Python function to parse a CSV file."}]
  }'
```

### Smart Routing Response Headers

| Header | Meaning |
|---|---|
| `X-Route-Decision` | Selected endpoint name |
| `X-Route-Confidence` | Routing confidence |
| `X-Route-Reason` | Human-readable routing reason |
| `X-Route-Strategy` | Classified workload type |
| `X-Route-GPU` / `X-Route-VRAM` / `X-Route-SOC` / `X-Route-CPU` / `X-Route-RAM` | Selected endpoint hardware metadata |
| `X-Route-Pinned` | `true` when an existing `X-Convo-ID` route pin was reused |
| `X-Route-Pin-Stale` | `true` when a previous route pin referenced an endpoint no longer in config and smart routing replaced it |

## Request Trace

Every successful proxied chat response includes:

| Header | Meaning |
|---|---|
| `X-Trace-ID` | Per-request trace identifier for correlating dashboard/runtime metadata |

The trace object is intentionally in-memory only in this phase; it is not persisted.

## Conversation History

The proxy stores conversation history and conversation metadata in local SQLite (`var/history.sqlite3` by default) only when `X-Convo-ID` is supplied. `X-Convo-ID` is the canonical cache/session key for persisted history, route pinning, reasoning effort, and optional slot affinity. Streamed assistant responses are read by a proxy-owned producer task, so upstream streaming and history finalization can continue even if the downstream client disconnects. Clients should send only the new turn for an existing conversation; full-history replay is rejected with `400`.

Example:

```bash
curl -X POST http://localhost:12340/smart \
  -H "Content-Type: application/json" \
  -H "X-Convo-ID: session-abc123" \
  -d '{
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'
```

Retrieve history:

```text
POST /conversations/retrieve
```

```bash
curl -X POST http://localhost:12340/conversations/retrieve \
  -H "Content-Type: application/json" \
  -d '{"convo_id": "session-abc123"}'
```

## Reasoning Control

Set:

```text
X-Reasoning-Effort: low | medium | high
```

The proxy pins the first valid reasoning effort per `X-Convo-ID`, injects a system message of the form `Reasoning: {value}`, and sets `reasoning_effort` in the request body. Later turns may omit the header and reuse the pinned value. Conflicting later values return `409`. Without `X-Convo-ID`, reasoning remains per-request.

## Best-Effort Slot Affinity

For llama.cpp endpoints, set `slot_affinity: true` on the endpoint config to let the proxy try to map `X-Convo-ID` to an upstream slot. The proxy probes `/slots?fail_on_no_slot=1`, persists the selected slot id in conversation metadata, and forwards `id_slot` plus `cache_prompt: true` when available. Slot affinity is best-effort: probe failures, non-llama upstreams, unavailable slots, or ignored `id_slot` fields do not fail the chat request.

Slot response headers:

| Header | Meaning |
|---|---|
| `X-Upstream-Slot-ID` | Slot id applied to the upstream request |
| `X-Upstream-Slot-Status` | `affinity-applied`, `unavailable`, or `skipped` |

## RAG Control

Set:

```text
X-RAG-Endpoint: http://localhost:8100/api/retrieve
```

When present, the proxy:

1. uses the latest real user turn as the retrieval query
2. POSTs `{query, limit}` to the selected retrieve endpoint, normalized to `/api/retrieve/context`
3. expects the RAG service to perform retrieval, ranking, thresholding, and grounding
4. rewrites only the latest user turn with the returned `grounded_user_message`

### RAG Response Headers

| Header | Meaning |
|---|---|
| `X-RAG-Endpoint` | Retrieve endpoint used |
| `X-RAG-Injected` | `true` if context was injected |
| `X-RAG-Hits` | Number of context blocks returned by the RAG service |
| `X-RAG-Mode` | Backend retrieval mode, if provided |
| `X-RAG-Truncated` | `true` if the grounded message was truncated by the backend |
| `X-RAG-Reason` | Failure/skip reason when no context was injected |

## Streaming and Non-Streaming

For SSE streaming, send `"stream": true`.

Streaming behavior:

- media type: `text/event-stream`
- proxy appends `stream_options.include_usage = true`
- reasoning is emitted as `delta.reasoning`

Non-streaming behavior:

- response is standard JSON
- reasoning is moved to `choices[0].message.reasoning`

## Model Discovery

Fetch all models across reachable endpoints:

```text
GET /models
```

```bash
curl http://localhost:12340/models
```

The response is OpenAI-compatible and includes endpoint metadata such as `endpoint`, `endpoint_url`, and hardware fields.

This endpoint also refreshes the router’s internal reachable-endpoint cache.

## Lightweight Search

Search discovery is available when `search.enabled: true` in `config.yaml`.

```text
POST /search/web
```

Example:

```bash
curl -X POST http://localhost:12340/search/web \
  -H "Content-Type: application/json" \
  -d '{
    "query": "qwen mtp llama.cpp",
    "context": "optional compact planner context",
    "provider": "auto",
    "count": 5,
    "freshness": "week"
  }'
```

Request fields:

- `query`: required search request
- `context`: optional compact context for the search planner
- `provider`: provider id or `auto`
- `count`: optional result count
- `freshness`: optional freshness hint such as `day`, `week`, or `month`

Behavior:

- may plan one or more provider-ready queries through a bounded SearchPlanner when configured
- may run async provider fanout when the planner returns multiple queries
- sends one HTTP request to each selected search provider in priority order per planned query
- parses only the returned SERP payload
- never fetches result pages or executes JavaScript
- returns normalized result candidates plus degradation warnings
- may include `original_query` and filtered `planner` metadata, including `queries`, when planning ran
- includes `wrapped_results`, a JSON string marked as untrusted for downstream LLM use

TL;DR: the proxy API is OpenAI-compatible plus control headers for smart routing, conversation history, reasoning, RAG retrieval, and an optional lightweight search-discovery endpoint.
