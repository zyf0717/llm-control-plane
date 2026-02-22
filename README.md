# llm-control-plane

A self-hosted LLM orchestration layer that sits in front of multiple local or remote LLM endpoints. It exposes an OpenAI-compatible HTTP API and intelligently dispatches requests based on workload type, endpoint availability, and configurable hardware preferences.

---

## Table of Contents

- [llm-control-plane](#llm-control-plane)
  - [Table of Contents](#table-of-contents)
  - [Architecture](#architecture)
  - [Components](#components)
  - [Configuration](#configuration)
  - [Using the Orchestrator](#using-the-orchestrator)
    - [1. Direct Endpoint Routing](#1-direct-endpoint-routing)
    - [2. Smart / Auto Routing](#2-smart--auto-routing)
    - [3. Multi-Turn Conversations](#3-multi-turn-conversations)
    - [4. Reasoning Effort Control](#4-reasoning-effort-control)
    - [5. Streaming vs Non-Streaming](#5-streaming-vs-non-streaming)
    - [6. Model Discovery](#6-model-discovery)
    - [7. Conversation History Retrieval](#7-conversation-history-retrieval)
  - [Routing Logic](#routing-logic)
  - [Reasoning Channel Extraction](#reasoning-channel-extraction)
  - [Dashboard](#dashboard)
  - [RAG Server](#rag-server)
  - [Authentication](#authentication)
  - [Setup](#setup)

---

## Architecture

```
Client
  │
  ▼
Orchestrator Proxy  (port 12340)
  ├── POST /smart          ← LLM-classified auto routing
  ├── POST /{endpoint}     ← Direct endpoint routing
  ├── GET  /models         ← Aggregated model list
  └── POST /conversations/retrieve
  │
  ├── LLM Router
  │     ├── WorkloadClassifier  (LLM-based)
  │     └── Workload preferences from config.yaml
  │
  └── Upstream LLM Endpoints (per config.yaml)

Dashboard  (port 12341)   ← Shiny for Python UI
RAG Server                ← Optional, in-memory vector search
```

---

## Components

| Component | Path | Description |
|---|---|---|
| Proxy | `src/orchestrator/proxy.py` | FastAPI app; handles all routing modes |
| LLM Router | `src/orchestrator/llm_router.py` | Workload classification and endpoint selection |
| Dashboard | `src/dashboard/` | Shiny for Python web UI |
| RAG Server | `src/rag/server.py` | In-memory document store with vector search |
| Entry point | `llm_control_plane.py` | Starts proxy and dashboard in parallel threads |

---

## Configuration

Endpoints and workload preferences are defined in `config.yaml`.

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

**Workload types:**

| Type | When it is used |
|---|---|
| `reasoning` | Multi-step analysis, trade-off comparisons, deep thinking tasks |
| `programming` | Writing, debugging, or explaining code |
| `tokens_per_second` | Long, detailed output with minimal reasoning or code |
| `ttft_content` | Simple Q&A, general chat, fast first-token tasks |
| `classification` | Internal — used to select the fastest endpoint for the LLM classifier itself |

Hardware fields (`gpu`, `vram`, `cpu`, `ram`, `soc`) are optional metadata attached to model list responses.

---

## Using the Orchestrator

The proxy listens on **port 12340** by default. All endpoints accept JSON bodies in OpenAI chat-completions format.

### 1. Direct Endpoint Routing

Send a request directly to a named endpoint defined in `config.yaml`. The proxy forwards everything verbatim to `{endpoint.url}/v1/chat/completions`.

```
POST http://localhost:12340/{endpoint-name}
```

```bash
curl -X POST http://localhost:12340/my-server \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.3-70b",
    "messages": [{"role": "user", "content": "Explain transformers briefly."}]
  }'
```

Use this when you know exactly which machine or model you want to target, regardless of availability or workload fit.

---

### 2. Smart / Auto Routing

The orchestrator analyses the latest user message, classifies its workload type using a fast LLM call, and selects the highest-priority available endpoint from `config.yaml`.

```
POST http://localhost:12340/smart
POST http://localhost:12340/          ← root also triggers smart routing
```

```bash
curl -X POST http://localhost:12340/smart \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a Python function to parse a CSV file."}]
  }'
```

The response includes routing metadata in HTTP headers:

| Header | Example value | Description |
|---|---|---|
| `X-Route-Decision` | `my-server` | Selected endpoint name |
| `X-Route-Confidence` | `0.9` | Routing confidence (0–1) |
| `X-Route-Reason` | `Selected preferred endpoint for programming workload` | Human-readable reason |
| `X-Route-Strategy` | `programming` | Classified workload type |
| `X-Route-GPU` | `NVIDIA GeForce RTX 4090` | Hardware spec (if available) |
| `X-Route-VRAM` | `24GB` | VRAM (if available) |
| `X-Route-SOC` | `Apple M4` | SoC (if available) |

**Routing fallback:** if the preferred endpoints for the classified workload are all unreachable, the router falls back to any reachable endpoint (confidence drops to 0.5). If no endpoints are reachable at all, a 500 is returned.

---

### 3. Multi-Turn Conversations

Pass a `X-Convo-ID` header to have the proxy accumulate conversation history server-side. Each request only needs to include the new user message; the proxy automatically prepends the full history before forwarding.

```bash
# First turn
curl -X POST http://localhost:12340/smart \
  -H "Content-Type: application/json" \
  -H "X-Convo-ID: session-abc123" \
  -d '{
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'

# Second turn — proxy includes the previous exchange automatically
curl -X POST http://localhost:12340/smart \
  -H "Content-Type: application/json" \
  -H "X-Convo-ID: session-abc123" \
  -d '{
    "messages": [{"role": "user", "content": "And what language do they speak there?"}]
  }'
```

The `X-Convo-ID` is echoed back in the response headers. Conversation state lives in-process; it is reset when the proxy restarts.

Works with both direct and smart routing endpoints.

---

### 4. Reasoning Effort Control

Set `X-Reasoning-Effort` to hint the model's reasoning depth. The proxy injects a system message and sets `reasoning_effort` in the request body.

```
X-Reasoning-Effort: low | medium | high
```

```bash
curl -X POST http://localhost:12340/my-server \
  -H "Content-Type: application/json" \
  -H "X-Reasoning-Effort: high" \
  -d '{
    "messages": [{"role": "user", "content": "Analyse the trade-offs between SQL and NoSQL databases."}]
  }'
```

The injected system message is `Reasoning: {low|medium|high}`. If a prior reasoning message is already in the conversation history (via `X-Convo-ID`), it is replaced with the new value.

Combining with smart routing and conversation tracking is fully supported:

```bash
curl -X POST http://localhost:12340/smart \
  -H "Content-Type: application/json" \
  -H "X-Convo-ID: session-xyz" \
  -H "X-Reasoning-Effort: medium" \
  -d '{
    "messages": [{"role": "user", "content": "Compare microservices vs monoliths."}]
  }'
```

---

### 5. Streaming vs Non-Streaming

The proxy supports both modes. Set `"stream": true` in the request body for SSE streaming, or omit / set `false` for a regular JSON response.

**Streaming:**

```bash
curl -X POST http://localhost:12340/smart \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Tell me a story."}],
    "stream": true
  }'
```

Streaming responses are forwarded as `text/event-stream`. The proxy automatically appends `"stream_options": {"include_usage": true}` so token usage is included in the final `[DONE]` chunk.

Reasoning content is separated into a `reasoning` field on each delta (see [Reasoning Channel Extraction](#reasoning-channel-extraction)) during streaming.

**Non-streaming:**

```bash
curl -X POST http://localhost:12340/smart \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Tell me a story."}]
  }'
```

Non-streaming responses are returned as standard JSON. Reasoning tags (`<think>…</think>` or `<|channel|>analysis…`) are extracted and placed in `choices[0].message.reasoning`, leaving `choices[0].message.content` with only the final answer.

---

### 6. Model Discovery

Fetch all models available across every reachable endpoint. Results include hardware specs and the originating endpoint name.

```
GET http://localhost:12340/models
```

```bash
curl http://localhost:12340/models
```

Response format (OpenAI-compatible):

```json
{
  "object": "list",
  "data": [
    {
      "id": "llama-3.3-70b",
      "object": "model",
      "owned_by": "my-server",
      "endpoint": "my-server",
      "endpoint_url": "https://my-llm-server.example.com",
      "gpu": "NVIDIA GeForce RTX 4090",
      "vram": "24GB"
    }
  ]
}
```

This endpoint also refreshes the internal `reachable_endpoints` cache used by smart routing, so calling it before heavy usage warms up the router.

---

### 7. Conversation History Retrieval

Retrieve the full accumulated message history for a conversation ID.

```
POST http://localhost:12340/conversations/retrieve
```

```bash
curl -X POST http://localhost:12340/conversations/retrieve \
  -H "Content-Type: application/json" \
  -d '{"convo_id": "session-abc123"}'
```

Returns a list of `{role, content}` message objects in chronological order.

---

## Routing Logic

Smart routing proceeds in three steps:

1. **Classification** — the latest user message is sent to the fastest available endpoint (determined by the `classification` workload preference list) with a compact system prompt. The model returns one of: `reasoning`, `programming`, `tokens_per_second`, `ttft_content`. If classification fails or times out, the workload defaults to `tokens_per_second`.

2. **Endpoint selection** — the router looks up the `endpoint_preference` list for the classified workload type and picks the first entry that is currently reachable (i.e. appeared in the last `/models` response).

3. **Fallback** — if no preferred endpoint is reachable, any reachable endpoint is used. If none are reachable, a 500 error is raised.

Classification uses a 10-second timeout and runs against `{fastest_endpoint}/v1/chat/completions` with `max_tokens: 20` and `temperature: 0.1`.

---

## Reasoning Channel Extraction

Some models embed chain-of-thought reasoning inline using special tokens. The proxy normalises two formats:

| Format | Start token | End token |
|---|---|---|
| Think tags | `<think>` | `</think>` |
| Channel tokens | `<\|channel\|>analysis<\|message\|>` | `<\|end\|><\|start\|>assistant<\|channel\|>final<\|message\|>` |

In **streaming** responses, content inside these regions is yielded with `delta.reasoning` instead of `delta.content`, leaving the content stream clean.

In **non-streaming** responses, the reasoning is moved to `choices[0].message.reasoning` and stripped from `choices[0].message.content`.

---

## Dashboard

A Shiny for Python web dashboard runs on **port 12341** and provides:

- Endpoint and model selection (including an **Auto** mode that uses smart routing)
- Streaming and non-streaming chat interface
- System prompt configuration
- Reasoning effort selector
- Conversation history viewer
- Display of routing decisions (`X-Route-*` headers) and token usage

The dashboard communicates exclusively through the proxy on port 12340.

---

## RAG Server

A lightweight FastAPI server (`src/rag/server.py`) provides:

- In-memory document store with add / delete / list operations
- Vector search using `sentence-transformers` (`all-MiniLM-L6-v2`)
- Keyword search fallback when the embedding model is unavailable
- Configurable `top_k` retrieval

The RAG server is independent of the proxy and dashboard and must be started separately.

---

## Authentication

Upstream endpoints are protected with Cloudflare Access service tokens. Set the following environment variables (or place them in a `.env` file):

```
API_KEY_ID=<CF-Access-Client-Id>
API_KEY_SECRET=<CF-Access-Client-Secret>
```

These are attached as `CF-Access-Client-Id` and `CF-Access-Client-Secret` headers on every upstream request. The dashboard also reads `PROXY_BASE_URL` to know where the proxy is running:

```
PROXY_BASE_URL=http://localhost:12340
```

---

## Setup

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate llm-control-plane

# Start the proxy and dashboard
python llm_control_plane.py
```

The proxy is available at `http://localhost:12340` and the dashboard at `http://localhost:12341`.

To run tests:

```bash
pytest
```
