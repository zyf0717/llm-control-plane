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

Smart routing uses the latest user message to classify workload and select a reachable endpoint:

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

The proxy stores conversation history in-process when `X-Convo-ID` is supplied.

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

1. uses the latest user turn as the retrieval query
2. POSTs to the selected retrieve endpoint
3. filters results by `min_confidence`
4. injects accepted chunks as a turn-local `system` message before the latest user turn

### RAG Response Headers

| Header | Meaning |
|---|---|
| `X-RAG-Endpoint` | Retrieve endpoint used |
| `X-RAG-Injected` | `true` if context was injected |
| `X-RAG-Confidence` | Top normalized confidence |
| `X-RAG-Distance` | Raw top distance when provided by the backend |
| `X-RAG-Threshold` | Configured injection threshold |
| `X-RAG-Hits` | Number of accepted chunks |
| `X-RAG-Method` | Backend retrieval method, if provided |
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

TL;DR: the proxy API is OpenAI-compatible plus control headers for smart routing, conversation history, reasoning, and RAG retrieval.
