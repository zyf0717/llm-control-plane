# Getting Started

## Setup

```bash
uv sync --locked
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first. Run commands
from the repository root. uv creates `.venv` using the Python 3.12 selection in
`.python-version`, installs the editable project, and includes development tools.
`pyproject.toml` declares dependencies; the checked-in `uv.lock` records exact versions.

For a runtime-only installation, use `uv sync --locked --no-dev`. To update a package,
run `uv lock --upgrade-package PACKAGE`, then `uv sync --locked`; check in the updated
lock file after validation. Use `uv add PACKAGE` to add a runtime dependency or
`uv add --dev PACKAGE` to add a development tool.

## Configuration Files

Create local runtime files from the checked-in examples:

```bash
cp config.example.yaml config.yaml
cp .env.example .env
```

## Environment Variables

Set these in your shell or `.env`:

```bash
API_KEY_ID=<CF-Access-Client-Id>
API_KEY_SECRET=<CF-Access-Client-Secret>
PROXY_BASE_URL=http://localhost:12340
HISTORY_DB_PATH=var/history.sqlite3
```

- `API_KEY_ID` and `API_KEY_SECRET` are attached to upstream requests.
- `PROXY_BASE_URL` is used by the dashboard to call the proxy.
- `HISTORY_DB_PATH` optionally overrides the default local SQLite history database path (`var/history.sqlite3`).
  Leave it unset or blank to use the default path.

## Run

```bash
uv run --locked python llm_control_plane.py
```

Runtime endpoints:

- Proxy: `http://localhost:12340` (binds `0.0.0.0`)
- Dashboard: `http://localhost:12341` (binds `127.0.0.1`)

## Test

```bash
uv run --locked pytest
```

## Where to Read Next

- [Architecture](architecture.md)
- [Configuration](configuration.md)
- [API Guide](api.md)
- [Dashboard Guide](dashboard.md)
- [Search Guide](search.md)
- [Workflow Guide](workflows.md)

Run `uv sync --locked`, set the Cloudflare and proxy environment variables, then start
the application with `uv run --locked python llm_control_plane.py`.
