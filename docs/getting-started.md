# Getting Started

## Setup

```bash
conda env create -f environment.yml
conda activate llm-control-plane
```

The conda environment installs the editable local project from `pyproject.toml`.

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
python llm_control_plane.py
```

Runtime endpoints:

- Proxy: `http://localhost:12340` (binds `0.0.0.0`)
- Dashboard: `http://localhost:12341` (binds `127.0.0.1`)

## Test

```bash
pytest
```

## Where to Read Next

- [Architecture](architecture.md)
- [Configuration](configuration.md)
- [API Guide](api.md)
- [Dashboard Guide](dashboard.md)
- [Search Guide](search.md)
- [Workflow Guide](workflows.md)

TL;DR: create the conda env, set the Cloudflare and proxy env vars, run `python llm_control_plane.py`, then use the focused docs for the operational details.
