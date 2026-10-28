# Workflow Guide

Workflows are YAML-defined multi-step runs loaded from `workflow_configs/`. They execute through the proxy runtime and can call configured LLM endpoints, search providers, RAG retrieval, and the dedicated reranker.

## Built-In Workflows

| Workflow | Purpose |
|---|---|
| `threaded_search` | Use thread context to plan searches, dispatch multiple queries, rerank globally, and answer the latest prompt |
| `threaded_rag` | Use thread context to plan multiple Retrieval queries, dispatch them concurrently, then answer from merged RAG evidence |
| `repo_context` | Explore one local repository through the configured repo-context CLI |
| `url_retrieval` | Fetch supplied URLs deterministically, then answer the original prompt using their content |

Research Brief and Implementation Plan have been retired. Use Search with the
desired answer format in the request, or Repo Context for repository-specific work.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/workflows` | List workflow summaries |
| `GET` | `/workflows/{workflow_id}` | Fetch full workflow spec |
| `POST` | `/workflows/{workflow_id}/runs` | Create a run |
| `GET` | `/workflow-runs?limit=50` | List recent runs |
| `DELETE` | `/workflow-runs` | Clear persisted workflow runs |
| `GET` | `/workflow-runs/{run_id}` | Fetch run snapshot |
| `POST` | `/workflow-runs/{run_id}/advance` | Execute one pending step |
| `POST` | `/workflow-runs/{run_id}/run` | Run to terminal state |
| `POST` | `/workflow-runs/{run_id}/run-stream` | Run to terminal state with SSE events |
| `POST` | `/workflow-runs/{run_id}/steps/{step_id}/retry` | Reset a failed/completed step and its dependents |

Create-run body:

```json
{
  "params": {"latest_user_prompt": "Find evidence for this feature"},
  "endpoint": "primary",
  "conversation_id": "optional-session",
  "reasoning_effort": "high",
  "retrieval_endpoint": "http://localhost:8100/api/retrieve/context",
  "search_provider": "duckduckgo_html"
}
```

`endpoint` is required for workflows containing `llm` or `compress_source` steps
and must be a concrete configured endpoint, not `smart`. Workflows containing
only deterministic steps can omit it. `url_retrieval` requires an endpoint for
its final answer step.

Model-backed steps can instead declare `workload_type: utility` to resolve their
endpoint from the [workload preference list](configuration.md#workload-preferences).
This takes precedence over the run endpoint. A step cannot specify both
`endpoint` and `workload_type`; steps with neither inherit the run endpoint.

LLM steps receive manual/uploaded sources and thread context as messages before
the rendered task prompt, including during streaming and JSON repair. The dashboard
passes history in `params.conversation_messages` as a list of `role`/`content`
messages so growing conversations retain stable cache boundaries. API callers can
pass the same list; `thread_briefing` is the text fallback. Avoid repeating these
context fields inside LLM prompt templates. Cache reuse depends on the upstream server.

## Spec Shape

```yaml
id: example
name: Example
version: 0.1.0
description: Optional summary

params_schema:
  type: object
  required: [goal]
  properties:
    goal:
      type: string

defaults:
  reasoning_effort: high
  stream: false
  max_tokens: 1024
  search_provider: duckduckgo_html
  retrieval_endpoint: http://localhost:8100/api/retrieve/context

steps:
  - id: plan
    kind: llm
    prompt: "{{ params.goal }}"
    output_key: plan
```

## Step Fields

Common fields:

| Field | Applies to | Meaning |
|---|---|---|
| `id` | all | Stable step id |
| `name` | all | Display label; defaults to `id` |
| `kind` | all | `llm`, `search`, `retrieval`, `rerank`, `manual`, `compress_source`, `repo_context`, `url_extract`, or `url_fetch` |
| `depends_on` | all | Step ids that must complete first |
| `prompt` | `llm`, `search`, `retrieval`, `rerank`, `url_extract`, `url_fetch` | Template rendered from params and prior outputs |
| `output_key` | all | Key used in `outputs`; defaults to step id |
| `chat_visibility` | all | `hidden`, `intermediate`, or `final` |
| `chat_stream` | `llm` | Override streaming for that LLM step |
| `endpoint` | `llm`, `compress_source` | Override the run endpoint for this model-backed step |
| `workload_type` | `llm`, `compress_source` | Resolve a configured workload instead of inheriting the run endpoint; mutually exclusive with `endpoint` |
| `max_tokens` | `llm` | Override max output tokens |
| `reasoning_effort` | `llm` | Override run/default reasoning effort |
| `retrieval_endpoint` | `llm`, `retrieval` | Override run/default RAG endpoint |
| `use_retrieval` | `llm`, `compress_source` | Set `false` to prevent a model-backed step from using the run/default RAG endpoint |

Search fields:

| Field | Meaning |
|---|---|
| `search_provider` | Override run/default provider |
| `search_count` | Provider result count per dispatched query; defaults to `5` |
| `use_query_refiner` | Whether the proxy query refiner may rewrite/fan out plain prompts |

Retrieval fields:

| Field | Meaning |
|---|---|
| `retrieval_count` | Context block limit per dispatched retrieval query; defaults to proxy retrieval config |

Rerank fields:

| Field | Meaning |
|---|---|
| `rerank_context` | Extra context sent to the reranker |
| `rerank_top_k` | Final reranked result cap |

Repo-context fields:

| Field | Meaning |
|---|---|
| `repo_context_repo` | Template resolving to one immediate child directory under configured `repos_root` |
| `repo_context_max_turns` | Optional CLI exploration turn cap; defaults to repo-context config |

Unsupported legacy fields:

- `use_reranker` on a `search` step is rejected. Add an explicit `rerank` step.
- `rerank_context` on a non-`rerank` step is rejected.
- `output_schema` is rejected. Use `output_contract`.

## Structured Output

LLM steps can use `output_contract` to parse and validate model output.

```yaml
output_contract:
  format: json
  required: true
  schema:
    type: object
    additionalProperties: false
    required: [queries]
    properties:
      queries:
        type: array
        minItems: 1
        items:
          type: string
  on_invalid:
    action: retry
    max_attempts: 2
    repair: true
```

JSON and YAML contracts persist both raw text and parsed JSON under the step output. Invalid output can be repaired and retried according to `on_invalid`.

## Search And Rerank Steps

A `search` step renders its prompt, extracts one or more queries, and dispatches them through the workflow search client. If the rendered prompt is JSON containing a string array, each unique string becomes a query. Multiple query results are deduped and merged; the merged output includes `queries`, `per_query`, `workflow_search`, and `results`.

A `retrieval` step follows the same query extraction shape as `search`: JSON arrays, `queries`, and `query` fields are accepted. It dispatches all extracted queries concurrently to the selected Retrieval endpoint, merges and dedupes returned context/evidence blocks, and persists `queries`, `per_query`, `workflow_retrieval`, `context_blocks`, `evidence_blocks`, and `grounded_user_messages`.

A `rerank` step selects a dependency output containing `results`, renders its own prompt as the ranking query, renders `rerank_context`, and calls the workflow reranker. The output overwrites `output_key` when configured, so downstream LLM steps can consume reranked results through the same logical key.

## Threaded Search Pattern

`threaded_search` uses three distinct query concepts:

| Value | Source | Purpose |
|---|---|---|
| `params.latest_user_prompt` | Dashboard or API run params | Original latest user turn |
| `outputs.search_plan.json.queries` | Planner JSON | Provider-facing search queries, up to five |
| `outputs.search_plan.json.rerank_query` | Planner JSON | Context-resolved information need used to rank documents |

The workflow currently sets:

- `search_count: 20` on the search step
- `rerank_top_k: 10` on the rerank step
- `use_query_refiner: false` because the planner owns query generation
- `endpoint` on the planning LLM step when a smaller/faster concrete endpoint is preferred

This keeps search recall and reranking relevance separate: provider queries can be broad and varied, while the reranker receives one resolved information need plus context.

## Threaded RAG Pattern

`threaded_rag` first runs `plan_rag_prompt` with `use_retrieval: false`. That step resolves pronouns, ellipses, omitted subjects, and thread-local constraints into one self-contained `retrieval_prompt` plus one to six `retrieval_queries`. It emits multiple queries when the prompt mentions, compares, or implies multiple entities, projects, repos, products, services, dates, or evidence angles.

The `retrieve_evidence` step dispatches those queries concurrently to the selected run-level Retrieval endpoint and merges the returned evidence. The final `answer_with_rag` LLM step also sets `use_retrieval: false`; it synthesizes from the merged retrieval output instead of triggering an additional single-query proxy Retrieval pass.

## URL Retrieval Pattern

`url_retrieval` exposes deterministic `url_extract` and `url_fetch` steps that can
also be used in custom workflows. `url_extract` consumes text and returns
`json.urls`; `url_fetch` consumes a JSON array of URL strings and returns
`json.results`. See [URL Retrieval](url-retrieval.md) for invocation, limits, and
content extraction behavior.

After fetching, `answer_from_urls` uses the selected model to follow the original
prompt, including its requested output format, with the source documents and
normal thread/supplied context. The discovery and fetch steps do not call an LLM.

## Dashboard Integration

The Workflows tab can create runs, advance one step, run to completion, retry ended steps, upload UTF-8 context files, and inspect artifacts. Single-Slot Workflow Dispatch can dispatch a chat turn into a selected workflow; `threaded_search` requires a concrete search provider, `threaded_rag` requires a concrete Retrieval endpoint, and other workflows default the Single-Slot provider selector back to `None`.

For `repo_context`, the dashboard lists immediate child directories under the configured `repo_context.repos_root`. Single-Slot dispatch maps the latest user turn to `query` and the selected repository to `repo_name`. The built-in workflow first plans a focused repo-context query, then runs hidden repository exploration, then returns a final repo-grounded answer. In the Workflows tab, the repository selector fills `repo_name` only when the editable params JSON leaves it missing or blank.

TL;DR: workflows are explicit DAGs. Use `search` for candidate collection, `rerank` for ranking, and `llm` steps for planning and synthesis.
