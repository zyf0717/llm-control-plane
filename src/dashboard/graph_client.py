from __future__ import annotations

import json
import logging
import os
from typing import Any, AsyncIterator

import httpx

PROXY_BASE_URL = os.getenv("PROXY_BASE_URL")
API_KEY_ID = os.getenv("API_KEY_ID")
API_KEY_SECRET = os.getenv("API_KEY_SECRET")
logger = logging.getLogger(__name__)


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if API_KEY_ID and API_KEY_SECRET:
        headers["X-API-Key-ID"] = API_KEY_ID
        headers["X-API-Key-Secret"] = API_KEY_SECRET
    return headers


async def fetch_graphs() -> list[dict[str, Any]]:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{PROXY_BASE_URL}/graphs", headers=_headers())
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        logger.warning("Failed to fetch graphs: %s", exc)
        return []
    graphs = data.get("graphs") if isinstance(data, dict) else []
    return [item for item in graphs if isinstance(item, dict)]


async def fetch_graph(graph_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{PROXY_BASE_URL}/graphs/{graph_id}", headers=_headers()
        )
        _raise_for_status(response)
        return response.json()


async def create_graph_run(graph_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            f"{PROXY_BASE_URL}/graphs/{graph_id}/runs",
            headers=_headers(),
            json=payload,
        )
        _raise_for_status(response)
        return response.json()


async def fetch_graph_runs(limit: int = 50) -> list[dict[str, Any]]:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{PROXY_BASE_URL}/graph-runs",
                headers=_headers(),
                params={"limit": limit},
            )
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        logger.warning("Failed to fetch graph runs: %s", exc)
        return []
    runs = data.get("runs") if isinstance(data, dict) else []
    return [item for item in runs if isinstance(item, dict)]


async def fetch_graph_run(run_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{PROXY_BASE_URL}/graph-runs/{run_id}", headers=_headers()
        )
        _raise_for_status(response)
        return response.json()


async def run_graph_to_completion(run_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{PROXY_BASE_URL}/graph-runs/{run_id}/run",
            headers=_headers(),
        )
        _raise_for_status(response)
        return response.json()


async def stream_graph_run_events(
    run_id: str,
    *,
    stream_mode: Any = None,
) -> AsyncIterator[dict[str, Any]]:
    payload = {}
    if stream_mode is not None:
        payload["stream_mode"] = stream_mode
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            f"{PROXY_BASE_URL}/graph-runs/{run_id}/stream",
            headers=_headers(),
            json=payload,
        ) as response:
            _raise_for_status(response)
            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    block, buffer = buffer.split("\n\n", 1)
                    event = _decode_sse_event(block)
                    if event is not None:
                        yield event


def _decode_sse_event(block: str) -> dict[str, Any] | None:
    data_lines = []
    for line in block.splitlines():
        if line.startswith("data: "):
            data_lines.append(line[6:])
    if not data_lines:
        return None
    try:
        value = json.loads("\n".join(data_lines))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _raise_for_status(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = ""
        try:
            payload = response.json()
            if isinstance(payload, dict):
                detail = str(payload.get("detail") or "")
        except Exception:
            detail = response.text
        message = detail or str(exc)
        raise RuntimeError(message) from exc

