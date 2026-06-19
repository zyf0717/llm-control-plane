# Graph Guide

Graphs are LangGraph-native orchestration units that live independently from YAML workflows. They use Python graph code and the LangGraph app convention instead of the workflow step-DAG YAML schema.

## Configuration

Graph refs are loaded from `langgraph.json`:

```json
{
  "dependencies": ["."],
  "graphs": {
    "research_agent": "./src/graphs/research_agent.py:graph"
  },
  "env": ".env"
}
```

Optional graph metadata lives in `graph_configs/{graph_id}.yaml`:

```yaml
id: research_agent
name: Research Agent
description: LangGraph-native research graph
input_schema:
  type: object
  required: [question]
  properties:
    question:
      type: string
defaults:
  configurable:
    endpoint: primary
    search_provider: duckduckgo_html
ui:
  supports_streaming: true
  supports_interrupts: true
```

Graph metadata is for the control plane and dashboard. Graph control flow belongs in Python LangGraph code.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/graphs` | List graph summaries |
| `GET` | `/graphs/{graph_id}` | Fetch graph metadata |
| `POST` | `/graphs/{graph_id}/runs` | Create a graph run |
| `GET` | `/graph-runs?limit=50` | List recent graph runs |
| `GET` | `/graph-runs/{run_id}` | Fetch run snapshot |
| `POST` | `/graph-runs/{run_id}/run` | Invoke the graph to completion |
| `POST` | `/graph-runs/{run_id}/stream` | Stream graph events over SSE |
| `POST` | `/graph-runs/{run_id}/resume` | Resume an interrupted graph |

Create-run body:

```json
{
  "input": {"question": "What changed?"},
  "config": {
    "configurable": {
      "thread_id": "optional-thread",
      "endpoint": "primary",
      "search_provider": "duckduckgo_html",
      "retrieval_endpoint": "local-retrieval"
    }
  }
}
```

If `config.configurable.thread_id` is omitted, the control plane uses the graph run id as the thread id.

## Separation From Workflows

Graphs and workflows share proxy runtime services such as endpoint routing, search, rerank, retrieval, tracing, auth, and logging. They do not share registries, stores, routes, request schemas, response schemas, dashboard code, or execution semantics.

Use workflows for explicit YAML step runs with `/advance` and step retry. Use graphs for LangGraph-native Python control flow, streaming, interrupts, and graph checkpoints.

