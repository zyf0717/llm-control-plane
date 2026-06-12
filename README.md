# llm-control-plane

An open-source control plane for exploratory work around how LLMs, tools, retrieval, search, and multi-step workflows should function together. It sits in front of multiple local or remote LLM endpoints, exposes an OpenAI-compatible HTTP API, and provides a Shiny dashboard for interactive experiments.

The repo is intentionally pragmatic rather than framework-heavy: it keeps routing, conversation state, RAG injection, ad hoc search, workflow execution, and observability visible enough to inspect and change. Auto routing is intended for stateless, non-agentic requests; dashboard conversations that have a conversation id pin Auto to the first selected concrete endpoint.

## Documentation

- [Docs Index](docs/README.md)
- [Getting Started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [API Guide](docs/api.md)
- [Dashboard Guide](docs/dashboard.md)

## Quick Start

```bash
conda env create -f environment.yml
conda activate llm-control-plane
python llm_control_plane.py
```

`environment.yml` bootstraps the editable local package from `pyproject.toml`.

- Proxy: `http://localhost:12340`
- Dashboard: `http://localhost:12341`

## Repo Layout

| Path | Purpose |
|---|---|
| `src/orchestrator/` | FastAPI app composition, request processing, upstream proxying, smart routing, workflow APIs |
| `src/dashboard/` | Shiny UI plus extracted search/workflow/trace server helpers |
| `src/search/` | Provider routing plus optional ad hoc query refinement |
| `workflow_configs/` | Context-driven workflow definitions |
| `config.yaml` | Local endpoint, routing, RAG, and search configuration |
| `config.example.yaml` | Checked-in configuration template |
| `.env.example` | Checked-in environment variable template |
| `llm_control_plane.py` | Starts proxy and dashboard together |
| `docs/` | Operational and architectural documentation |

## Development

```bash
pytest
```

Project dependencies now live in `pyproject.toml`; the conda file is only a thin wrapper for local env creation.

Ad hoc Single-Node search may use the query refiner to produce provider-ready search query/queries. Workflow search keeps query planning inside workflow LLM steps and bypasses the query refiner during dispatch.
