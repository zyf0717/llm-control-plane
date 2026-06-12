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
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}


async def create_workflow_run(workflow_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{PROXY_BASE_URL}/workflows/{workflow_id}/runs",
            headers=_headers(),
            json=payload,
        )
        response.raise_for_status()
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
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}


async def advance_workflow_run(run_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{PROXY_BASE_URL}/workflow-runs/{run_id}/advance", headers=_headers()
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}


async def run_workflow_to_completion(run_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(
            f"{PROXY_BASE_URL}/workflow-runs/{run_id}/run",
            headers=_headers(),
            json={},
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}


async def retry_workflow_step(run_id: str, step_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{PROXY_BASE_URL}/workflow-runs/{run_id}/steps/{step_id}/retry",
            headers=_headers(),
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}
