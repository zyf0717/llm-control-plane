# llm-control-plane

A self-hosted LLM orchestration layer that sits in front of multiple local or remote LLM endpoints. It exposes an OpenAI-compatible HTTP API, supports workload-aware routing for isolated tasks, and provides a Shiny dashboard for interactive use. Auto routing is intended for stateless, non-agentic requests; without `X-Convo-ID` there is no proxy-persisted conversation history, and dashboard conversations that do have a conversation id pin Auto to the first selected concrete endpoint.

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
| `src/orchestrator/` | FastAPI proxy, smart router, request/response shaping |
| `src/dashboard/` | Shiny dashboard UI and server logic |
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

TL;DR: the root README is now the entry point; the detailed docs live under [`docs/`](docs/README.md).
