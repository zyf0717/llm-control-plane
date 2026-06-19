# Workflow Guide

Workflows are YAML-defined multi-step runs loaded from `workflow_configs/`. They execute through the proxy runtime and can call configured LLM endpoints, search providers, RAG retrieval, and the dedicated reranker.

## Built-In Workflows

| Workflow | Purpose |
|---|---|
| `contextual_search` | Use conversation context to plan searches, dispatch multiple queries, rerank globally, and answer the latest prompt |
| `research_brief` | Build a concise brief from a question, optional manual context, uploaded files, search, and rerank |
| `implementation_plan` | Build staged implementation guidance from a goal, optional context, search, and rerank |
| `repo_context` | Explore one local repository through the configured repo-context CLI |

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
  "params": {"goal": "ship the feature"},
  "endpoint": "primary",
  "conversation_id": "optional-session",
  "reasoning_effort": "high",
  "retrieval_endpoint": "http://localhost:8100/api/retrieve/context",
  "search_provider": "duckduckgo_html"
}
```

`endpoint` is required and must be a concrete configured endpoint, not `smart`.

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
| `kind` | all | `llm`, `search`, `rerank`, `manual`, `compress_source`, or `repo_context` |
| `depends_on` | all | Step ids that must complete first |
| `prompt` | `llm`, `search`, `rerank` | Template rendered from params and prior outputs |
| `output_key` | all | Key used in `outputs`; defaults to step id |
| `chat_visibility` | `llm` | `hidden`, `intermediate`, or `final` |
| `chat_stream` | `llm` | Override streaming for that LLM step |
| `endpoint` | `llm`, `compress_source` | Override the run endpoint for this model-backed step |
| `max_tokens` | `llm` | Override max output tokens |
| `reasoning_effort` | `llm` | Override run/default reasoning effort |
| `retrieval_endpoint` | `llm` | Override run/default RAG endpoint |

Search fields:

| Field | Meaning |
|---|---|
| `search_provider` | Override run/default provider |
| `search_count` | Provider result count per dispatched query; defaults to `5` |
| `use_query_refiner` | Whether the proxy query refiner may rewrite/fan out plain prompts |

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

A `rerank` step selects a dependency output containing `results`, renders its own prompt as the ranking query, renders `rerank_context`, and calls the workflow reranker. The output overwrites `output_key` when configured, so downstream LLM steps can consume reranked results through the same logical key.

## Contextual Search Pattern

`contextual_search` uses three distinct query concepts:

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

## Dashboard Integration

The Workflows tab can create runs, advance one step, run to completion, retry ended steps, upload UTF-8 context files, and inspect artifacts. Single-Node Workflow Dispatch can dispatch a chat turn into a selected workflow; `contextual_search` requires a concrete search provider, while non-search workflows default the Single-Node provider selector back to `None`.

For `repo_context`, the dashboard lists immediate child directories under the configured `repo_context.repos_root`. Single-Node dispatch maps the latest user turn to `query` and the selected repository to `repo_name`. The built-in workflow first plans a focused repo-context query, then runs hidden repository exploration, then returns a final repo-grounded answer. In the Workflows tab, the repository selector fills `repo_name` only when the editable params JSON leaves it missing or blank.

TL;DR: workflows are explicit DAGs. Use `search` for candidate collection, `rerank` for ranking, and `llm` steps for planning and synthesis.
