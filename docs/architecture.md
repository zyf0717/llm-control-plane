# Architecture

## Components

| Component | Path | Responsibility |
|---|---|---|
| Proxy | `src/orchestrator/proxy.py` | OpenAI-compatible request handling, history, reasoning injection, RAG injection, upstream proxying |
| LLM Router | `src/orchestrator/llm_router.py` | Workload classification and smart endpoint selection |
| Dashboard | `src/dashboard/` | Shiny UI for chat, endpoint selection, RAG selection, and runtime inspection |
| Entry point | `llm_control_plane.py` | Runs proxy and dashboard together |

## End-to-End Flow

```mermaid
flowchart TD
    U[User or Client] -->|HTTP chat request| P[Proxy :12340]
    U -->|Browser UI| D[Dashboard :12341]

    D -->|GET /models| P
    D -->|GET health_url| RH[RAG health endpoints]
    D -->|POST /smart or /{endpoint}<br/>X-Convo-ID<br/>X-Reasoning-Effort<br/>X-RAG-Endpoint| P

    P -->|GET /v1/models| M[Configured LLM endpoints]
    M -->|Model metadata| P

    P -->|Latest user turn| R[LLM Router]
    R -->|Workload classification| P

    P -->|POST retrieve_url| RR[Selected RAG endpoint]
    RR -->|Retrieved chunks| P

    P -->|Forward final messages| LLM[Upstream LLM endpoint]
    LLM -->|SSE or JSON response| P

    P -->|Normalized response + X-Route-* + X-RAG-*| D
    P -->|Normalized response + X-Route-* + X-RAG-*| U
```

## Request Processing

The proxy processes chat requests in this order:

1. Parse the incoming body and headers.
2. Apply conversation history if `X-Convo-ID` is present.
3. Inject reasoning control if `X-Reasoning-Effort` is present.
4. Retrieve RAG context if `X-RAG-Endpoint` is present.
5. For smart routing, classify the latest user turn and select the best reachable endpoint.
6. Forward the final request to the selected upstream model endpoint.
7. Normalize streaming/non-streaming reasoning content and attach response metadata headers.

## Smart Routing Logic

Smart routing proceeds in three steps:

1. Classification: the latest user message is sent to the fastest available endpoint from the `classification` workload preference list.
2. Endpoint selection: the first reachable endpoint in the selected workload’s `endpoint_preference` list is chosen.
3. Fallback: if none of the preferred endpoints are reachable, any reachable endpoint is used; if none are reachable, the proxy returns `500`.

Classification uses:

- timeout: `10s`
- `max_tokens: 20`
- `temperature: 0.1`

## Reasoning Channel Extraction

The proxy normalizes these reasoning formats:

| Format | Start token | End token |
|---|---|---|
| Think tags | `<think>` | `</think>` |
| Channel tokens | `<\|channel\|>analysis<\|message\|>` | `<\|end\|><\|start\|>assistant<\|channel\|>final<\|message\|>` |

- In streaming mode, reasoning is emitted as `delta.reasoning`.
- In non-streaming mode, reasoning is moved to `choices[0].message.reasoning`.

## RAG Injection Model

RAG context is not merged into the dashboard system prompt field. It is injected by the proxy as a turn-local `system` message immediately before the latest user turn.

That preserves:

- system prompt ownership for stable behavioral instructions
- turn-local retrieval semantics
- conversation history without stale retrieved context

For backends that return `score`, that value is used directly as confidence. For backends that return `distance`, confidence is normalized as `1.0 - distance` before thresholding.

TL;DR: the proxy is the integration point; the dashboard selects endpoints and shows metadata, but the proxy owns history, routing, reasoning, and RAG injection.
