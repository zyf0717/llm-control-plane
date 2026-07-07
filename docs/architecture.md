# Architecture

## Components

| Component | Path | Responsibility |
|---|---|---|
| Proxy app | `src/orchestrator/proxy.py` | FastAPI app composition, public routes, model discovery, router inclusion |
| Request processing | `src/orchestrator/request_processor.py` | Conversation history/state, reasoning injection, RAG injection, slot affinity |
| Upstream proxying | `src/orchestrator/upstream_proxy.py` | Streaming/non-streaming upstream calls, response normalization, trace/history finalization |
| Runtime services | `src/orchestrator/proxy_services.py` | Configured endpoints, search service, history store, orchestration subsystem startup/shutdown |
| Orchestration boundary | `src/orchestrator/orchestration/` | Subsystem lifecycle and router interface |
| Runtime adapters | `src/orchestrator/runtime/` | Proxy-backed LLM/search adapters shared by orchestration subsystems |
| Search API | `src/orchestrator/search_routes.py` | `/search/web` request parsing and response wrapping |
| Search core | `src/search/` | Provider selection, query refinement, result normalization, dedupe, optional reranking |
| Workflow API | `src/orchestrator/workflow/api.py` | Workflow catalog and run lifecycle routes |
| Workflow executor | `src/orchestrator/workflow/` | YAML workflow model validation, DAG execution, output contracts |
| Graph API | `src/orchestrator/graph/api.py` | LangGraph catalog and run lifecycle routes |
| Graph runtime | `src/orchestrator/graph/` | LangGraph registry, run store, executor, and subsystem wrapper |
| LLM Router | `src/orchestrator/llm_router.py` | Workload classification and smart endpoint selection |
| Dashboard | `src/dashboard/` | Shiny UI for chat, endpoint selection, RAG selection, and runtime inspection |
| Entry point | `llm_control_plane.py` | Runs proxy and dashboard together |

## End-to-End Flow

```mermaid
flowchart TD
    U[User or Client] -->|HTTP chat request| P[Proxy :12340]
    U -->|Browser UI| D[Dashboard :12341 localhost]

    D -->|GET /models| P
    D -->|GET health_url| RH[RAG health endpoints]
    D -->|POST /smart or /{endpoint}<br/>X-Convo-ID<br/>X-Reasoning-Effort<br/>X-Allow-*-Switch<br/>X-RAG-Endpoint| P
    D -->|POST /search/web| P
    D -->|Workflow API / Graph API| P

    P -->|GET /v1/models| M[Configured LLM endpoints]
    M -->|Model metadata| P

    P -->|Latest user turn| R[LLM Router]
    R -->|Workload classification| P

    P -->|POST retrieve_url| RR[Selected RAG endpoint]
    RR -->|Retrieved chunks| P
    P -->|Search route| S[Search service]
    S -->|Provider queries| SP[Search providers]
    S -->|Optional explicit rerank| RK[Reranker]
    P -->|Workflow routes| W[Workflow subsystem]
    W -->|LLM/search/rerank steps| P
    P -->|Graph routes| G[Graph subsystem]
    G -->|LangGraph nodes| P

    P -->|Forward final messages| LLM[Upstream LLM endpoint]
    LLM -->|SSE or JSON response| P

    P -->|Normalized response + X-Route-* + X-RAG-*| D
    P -->|Normalized response + X-Route-* + X-RAG-*| U
```

## Request Processing

The proxy processes chat requests in this order:

1. Parse the incoming body and headers.
2. Atomically resolve, pin, or explicitly switch conversation route/reasoning metadata when `X-Convo-ID` is present.
3. Load persisted conversation history, reject full-history replay, then append only the current durable client messages.
4. Inject the effective reasoning control.
5. Retrieve RAG context if `X-RAG-Endpoint` is present.
6. For smart routing, reuse a valid route pin or classify the latest user turn and pin the selected endpoint.
7. Apply best-effort llama.cpp slot affinity for endpoints with `slot_affinity: true`.
8. Forward the final request to the selected upstream model endpoint.
9. Normalize streaming/non-streaming reasoning content, persist assistant text when present, and attach response metadata headers, including `X-Trace-ID`. Streaming uses a proxy-owned upstream producer and a downstream client consumer, so assistant accumulation and history finalization can continue after client disconnect. Pending producer/finalization tasks are drained on shutdown.

## Search Model

`POST /search/web` is a lightweight candidate-discovery API. Query refinement is enabled by default when configured; reranking is disabled by default and requires `use_reranker: true`. The dashboard Single-Node ad hoc path keeps this lightweight by sending `count: 5` and `use_reranker: false`.

Workflow search is different: a `search` step can dispatch one or more planned queries and a later `rerank` step can rank the merged candidates. The built-in `threaded_search` workflow plans up to five provider queries and reranks the merged candidates. Workflow Retrieval can also run as an explicit `retrieval` step; `threaded_rag` plans one to six thread-resolved retrieval queries, dispatches them concurrently, and synthesizes from merged context without an extra single-query proxy Retrieval pass.

## Workflow Model

Workflows are validated YAML DAGs loaded from `workflow_configs/`. Runs are persisted by the proxy runtime and can be advanced step-by-step, run to completion, streamed, retried, or cleared through the workflow API. Step kinds are `llm`, `search`, `retrieval`, `rerank`, `manual`, `compress_source`, and `repo_context`.

Workflow LLM calls use the same upstream proxy machinery as direct chat requests. Workflow search/rerank calls use proxy-backed clients, so provider config, query-refiner config, reranker config, and result metadata stay centralized.

## Graph Model

Graphs are LangGraph-native orchestration units loaded from `langgraph.json` with optional metadata beside graph code in `src/graphs/*.yaml`. They have separate `/graphs` and `/graph-runs` APIs, separate graph run tables, and a separate dashboard tab.

Graphs and workflows share neutral runtime adapters for LLM/search/rerank/retrieval access, but they do not share executor, registry, persistence schema, request schema, or dashboard feature code. Either subsystem can be disabled under `orchestration` without changing the other subsystem.

## Smart Routing Logic

Smart routing remains stateless without `X-Convo-ID`. With `X-Convo-ID`, the proxy treats the conversation id as canonical state: the first smart decision pins the route server-side, later `/smart` calls reuse the pin, and removed configured endpoints are reported as stale before rerouting. The dashboard does not keep a parallel smart-routing pin; direct dashboard endpoint selections opt in to proxy-managed route switching.

Smart routing proceeds in three steps:

1. Classification: the latest user message is sent to the fastest available endpoint from the `classification` workload preference list.
2. Endpoint selection: the first reachable endpoint in the selected workload’s `endpoint_preference` list is chosen.
3. Fallback: if none of the preferred endpoints are reachable, any reachable endpoint is used; if none are reachable, the proxy returns `500`.

Classification uses:

- timeout: `10s`
- `max_tokens: 20`
- `temperature: 0.1`

## Slot Affinity

Endpoint config may set `slot_affinity: true` for llama.cpp servers. The proxy probes `/slots?fail_on_no_slot=1`, stores `conversation_id -> endpoint -> slot_id`, and forwards `id_slot` plus `cache_prompt: true` when a slot is available. This is opportunistic only: non-llama upstreams, disabled slot endpoints, probe errors, and ignored slot fields do not fail the chat request.

## Reasoning Channel Extraction

The proxy normalizes these reasoning formats:

| Format | Start token | End token |
|---|---|---|
| Think tags | `<think>` | `</think>` |
| Channel tokens | `<\|channel\|>analysis<\|message\|>` | `<\|end\|><\|start\|>assistant<\|channel\|>final<\|message\|>` |

- In streaming mode, reasoning is emitted as `delta.reasoning`.
- In non-streaming mode, reasoning is moved to `choices[0].message.reasoning`.

## RAG Grounding Model

RAG context is not merged into the dashboard system prompt field. When `X-RAG-Endpoint` is set, the proxy sends the latest real user turn to the normalized `/api/retrieve/context` route with a `limit`, then rewrites only that latest user turn with the backend's returned `grounded_user_message`.

That preserves:

- system prompt ownership for stable behavioral instructions
- turn-local retrieval semantics
- conversation history without stale retrieved context

The RAG service owns retrieval, ranking, thresholding, and prompt assembly. The proxy reports endpoint, hit count, injection status, backend mode, truncation, and skip reason headers when available.

TL;DR: the proxy is the integration point; the dashboard selects endpoints and shows metadata, but the proxy owns history, routing, reasoning, RAG injection, search, and orchestration subsystem wiring.
