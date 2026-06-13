# Search Guide

Search is a candidate-discovery subsystem exposed through `POST /search/web` and reused by workflow search steps. It returns normalized candidate links and snippets; it does not fetch destination pages or execute JavaScript.

## Request Modes

| Path | Query source | Rerank behavior |
|---|---|---|
| Direct API `/search/web` | Request body `query`, optionally refined | Reranker is off by default; send `use_reranker: true` to enable |
| Dashboard Single-Node search | Latest user turn plus compact conversation context | Dashboard sends `count: 5` and `use_reranker: false` |
| Workflow `search` step | Rendered step prompt, possibly multiple planned queries | Search never reranks inline; use a separate `rerank` step |

## `/search/web`

Minimal request:

```json
{
  "query": "qwen mtp llama.cpp",
  "provider": "auto",
  "count": 5
}
```

Optional controls:

| Field | Default | Meaning |
|---|---:|---|
| `context` | `null` | Compact context for query refinement and optional reranking |
| `provider` | `auto` | Provider id, `auto`, empty, or omitted |
| `count` | `search.max_results` | Requested provider result count, capped by `search.max_results` |
| `freshness` | `null` | Freshness hint such as `day`, `week`, or `month` |
| `use_query_refiner` / `useQueryRefiner` | `true` | Bypass query refinement when false |
| `use_reranker` / `useReranker` | `false` | Enable post-retrieval reranking when true |
| `rerank_context` / `rerankContext` | `context` | Optional ranking-specific context |

Responses include normalized `results`, `warnings`, `degraded`, and `wrapped_results`. When query refinement runs, response metadata may include `original_query` and `query_refinement`. When reranking runs, response metadata may include `reranking`.

## Query Refinement

If `search.query_refiner_model_endpoint` is configured and enabled, `/search/web` may convert a user query into provider-ready queries. `query_refiner_max_queries` bounds the fanout. If the refiner returns multiple queries, the router searches each query and dedupes results up to `search.search_max_total_results`.

Workflow-planned search usually sets `use_query_refiner: false` because the workflow LLM already produced provider-facing queries.

## Reranking

Reranking is opt-in for direct `/search/web` calls. The proxy default is `use_reranker: false`.

Dedicated reranker backend request shape:

```json
{
  "query": "context-resolved ranking query",
  "documents": ["Title\nSnippet\nURL"],
  "top_k": 10
}
```

The parser accepts common dedicated reranker response envelopes:

- `{ "scores": [...] }`
- `{ "results": [...] }`
- `{ "ranked": [...] }`
- `{ "rankings": [...] }`
- `{ "data": [...] }`

Items may identify candidates by `index`, `id`, or position. Scores are normalized into the candidate `score` field when present.

## Candidate Limits

| Limit | Applies to |
|---|---|
| `search.max_results` | Per-provider normalized result cap |
| `search.search_max_total_results` | Multi-query query-refiner fanout dedupe cap |
| `search.reranker_max_candidates` | Max candidates sent to the reranker |
| Workflow `search_count` | Per-dispatched-query provider count for a workflow search step |
| Workflow `rerank_top_k` | Final result cap for a workflow rerank step |

For contextual workflows using a strong dedicated reranker, the current built-in pattern plans up to five search queries, requests up to twenty results per query, merges them, and reranks to ten final candidates.

## Dashboard Single-Node Search

The Single-Node tab search provider selector controls ad hoc search augmentation. When selected, the dashboard calls `/search/web`, renders a search-candidate preface, and injects one turn-local untrusted synthetic user message. That synthetic message is merged into the next real user turn by the proxy and is not persisted as durable conversation history.

Single-Node search intentionally keeps `count: 5` and disables inline reranking. Use workflow routing or the Workflows tab for broader search plus explicit reranking.

TL;DR: `/search/web` is lightweight candidate discovery. Rerank is explicit: request it directly with `use_reranker: true`, or model it as a workflow `rerank` step.
