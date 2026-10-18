# llm-control-plane

An open-source control plane for exploratory work around how LLMs, tools, retrieval, search, and multi-step workflows should function together. It sits in front of multiple local or remote LLM endpoints, exposes an OpenAI-compatible HTTP API, and provides a Shiny dashboard for interactive experiments.

The repo is intentionally pragmatic rather than framework-heavy: it keeps routing, conversation state, RAG injection, ad hoc search, workflow execution, and observability visible enough to inspect and change. Auto routing is intended for stateless, non-agentic requests; dashboard conversations that have a conversation id pin Auto to the first selected concrete endpoint.

## Documentation

- [Docs Index](docs/README.md)
- [Getting Started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- [Frontend Migration Plan](docs/frontend-migration.md)
- [Configuration](docs/configuration.md)
- [API Guide](docs/api.md)
- [Dashboard Guide](docs/dashboard.md)
- [Search Guide](docs/search.md)
- [Workflow Guide](docs/workflows.md)
- [Graph Guide](docs/graphs.md)

## Quick Start

```bash
uv sync --locked
uv run --locked python llm_control_plane.py
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first.

- Proxy: `http://localhost:12340` (binds `0.0.0.0`)
- Dashboard: `http://localhost:12341` (binds `127.0.0.1`)

## Repo Layout

| Path | Purpose |
|---|---|
| `src/orchestrator/` | FastAPI app composition, request processing, upstream proxying, smart routing, workflow and graph APIs |
| `src/dashboard/` | Shiny UI plus extracted search/workflow/graph/trace server helpers |
| `src/search/` | Provider routing, query refinement, and optional explicit reranking |
| `workflow_configs/` | Context-driven workflow definitions |
| `src/graphs/*.yaml` | Optional LangGraph graph metadata |
| `langgraph.json` | LangGraph graph refs loaded by the graph subsystem |
| `config.yaml` | Local endpoint, routing, RAG, and search configuration |
| `config.example.yaml` | Checked-in configuration template |
| `.env.example` | Checked-in environment variable template |
| `llm_control_plane.py` | Starts proxy and dashboard together |
| `docs/` | Operational and architectural documentation |

## Development

```bash
uv run --locked pytest
```

Project dependencies live in `pyproject.toml` and are locked in `uv.lock`. uv manages
the local `.venv`; `.python-version` selects Python 3.12. The default sync includes
the development dependency group.

Ad hoc Single-Slot search may use the query refiner but intentionally disables inline reranking. Direct `/search/web` callers can opt into reranking with `use_reranker: true`. Workflow search can either use the query refiner or workflow-planned queries; workflow reranking is always an explicit workflow step.
