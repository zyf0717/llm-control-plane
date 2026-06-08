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

Smart routing uses the latest user message to classify workload and select a reachable endpoint. It is intended for isolated, non-agentic tasks where the request itself contains all needed context. Without `X-Convo-ID`, the proxy does not persist conversation history. For multi-turn or agentic flows, call `/smart` once, persist `X-Route-Decision`, and send follow-up turns to that concrete endpoint.

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

## Conversation History

The proxy stores conversation history in a local SQLite database (`var/history.sqlite3` by default) only when `X-Convo-ID` is supplied.

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

The proxy injects a system message of the form `Reasoning: {value}` and also sets `reasoning_effort` in the request body.

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
