import json
import logging
import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
PROXY_BASE_URL = os.getenv("PROXY_BASE_URL")
API_KEY_ID = os.getenv("API_KEY_ID")
API_KEY_SECRET = os.getenv("API_KEY_SECRET")


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if API_KEY_ID:
        headers["CF-Access-Client-Id"] = API_KEY_ID
    if API_KEY_SECRET:
        headers["CF-Access-Client-Secret"] = API_KEY_SECRET
    return headers


async def fetch_workflows() -> list[dict[str, Any]]:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{PROXY_BASE_URL}/workflows", headers=_headers())
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        logger.warning("Failed to fetch workflows: %s", exc)
        return []
    workflows = data.get("workflows") if isinstance(data, dict) else []
    return [item for item in workflows if isinstance(item, dict)]


async def fetch_workflow(workflow_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{PROXY_BASE_URL}/workflows/{workflow_id}", headers=_headers()
        )
        _raise_for_status(response)
        data = response.json()
        return data if isinstance(data, dict) else {}


async def create_workflow_run(workflow_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{PROXY_BASE_URL}/workflows/{workflow_id}/runs",
            headers=_headers(),
            json=payload,
        )
        _raise_for_status(response)
        data = response.json()
        return data if isinstance(data, dict) else {}


async def fetch_workflow_runs(limit: int = 50) -> list[dict[str, Any]]:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{PROXY_BASE_URL}/workflow-runs",
                headers=_headers(),
                params={"limit": int(limit)},
            )
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        logger.warning("Failed to fetch workflow runs: %s", exc)
        return []
    runs = data.get("runs") if isinstance(data, dict) else []
    return [item for item in runs if isinstance(item, dict)]


async def fetch_workflow_run(run_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{PROXY_BASE_URL}/workflow-runs/{run_id}", headers=_headers()
        )
        _raise_for_status(response)
        data = response.json()
        return data if isinstance(data, dict) else {}


async def advance_workflow_run(run_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{PROXY_BASE_URL}/workflow-runs/{run_id}/advance", headers=_headers()
        )
        _raise_for_status(response)
        data = response.json()
        return data if isinstance(data, dict) else {}


async def run_workflow_to_completion(run_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(
            f"{PROXY_BASE_URL}/workflow-runs/{run_id}/run",
            headers=_headers(),
            json={},
        )
        _raise_for_status(response)
        data = response.json()
        return data if isinstance(data, dict) else {}


async def stream_workflow_run_events(
    run_id: str,
    *,
    stream: bool = True,
    max_steps: int | None = None,
):
    payload: dict[str, Any] = {"stream": bool(stream)}
    if max_steps is not None:
        payload["max_steps"] = int(max_steps)
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            f"{PROXY_BASE_URL}/workflow-runs/{run_id}/run-stream",
            headers=_headers(),
            json=payload,
        ) as response:
            _raise_for_status(response)
            async for event in _iter_sse_events(response):
                yield event


async def retry_workflow_step(run_id: str, step_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{PROXY_BASE_URL}/workflow-runs/{run_id}/steps/{step_id}/retry",
            headers=_headers(),
        )
        _raise_for_status(response)
        data = response.json()
        return data if isinstance(data, dict) else {}


def _raise_for_status(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = _response_error_detail(response)
        if detail:
            raise RuntimeError(detail) from exc
        raise


def _response_error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if detail:
            return str(detail)
    text = str(getattr(response, "text", "") or "").strip()
    return text


async def _iter_sse_events(response: httpx.Response):
    event_type = ""
    data_lines: list[str] = []
    async for raw_line in response.aiter_lines():
        line = raw_line.strip()
        if not line:
            if data_lines:
                event = _decode_sse_event(event_type, data_lines)
                if event is not None:
                    yield event
            event_type = ""
            data_lines = []
            continue
        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
    if data_lines:
        event = _decode_sse_event(event_type, data_lines)
        if event is not None:
            yield event


def _decode_sse_event(event_type: str, data_lines: list[str]) -> dict[str, Any] | None:
    data = "\n".join(data_lines).strip()
    if not data:
        return None
    try:
        event = json.loads(data)
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict):
        return None
    event.setdefault("type", event_type or "message")
    return event
