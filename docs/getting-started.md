# Getting Started

## Setup

```bash
conda env create -f environment.yml
conda activate llm-control-plane
```

## Environment Variables

Set these in your shell or `.env`:

```bash
API_KEY_ID=<CF-Access-Client-Id>
API_KEY_SECRET=<CF-Access-Client-Secret>
PROXY_BASE_URL=http://localhost:12340
```

- `API_KEY_ID` and `API_KEY_SECRET` are attached to upstream requests.
- `PROXY_BASE_URL` is used by the dashboard to call the proxy.

## Run

```bash
python llm_control_plane.py
```

Runtime endpoints:

- Proxy: `http://localhost:12340`
- Dashboard: `http://localhost:12341`

## Test

```bash
pytest
```

## Where to Read Next

- [Architecture](architecture.md)
- [Configuration](configuration.md)
- [API Guide](api.md)
- [Dashboard Guide](dashboard.md)

TL;DR: create the conda env, set the Cloudflare and proxy env vars, run `python llm_control_plane.py`, then use the focused docs for the operational details.
