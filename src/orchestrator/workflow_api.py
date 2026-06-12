from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Request

from .workflow_executor import WorkflowExecutor
from .workflow_registry import WorkflowRegistry
from .workflow_store import SQLiteWorkflowStore


def create_workflow_router(
    *,
    registry_getter: Callable[[], WorkflowRegistry],
    store_getter: Callable[[], SQLiteWorkflowStore],
    executor_getter: Callable[[], WorkflowExecutor],
) -> APIRouter:
    router = APIRouter()

    @router.get("/workflows")
    async def list_workflows():
        registry = registry_getter()
        return {"workflows": [spec.summary_dict() for spec in registry.list()]}

    @router.get("/workflows/{workflow_id}")
    async def get_workflow(workflow_id: str):
        try:
            return registry_getter().get(workflow_id).to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/workflows/{workflow_id}/runs")
    async def create_run(workflow_id: str, request: Request):
        body = await _json_body(request)
        try:
            snapshot = await executor_getter().create_run(
                workflow_id,
                params=_dict_or_empty(body.get("params")),
                convo_id=_optional_str(body.get("convo_id")),
                endpoint=_optional_str(body.get("endpoint")),
                reasoning_effort=_optional_str(body.get("reasoning_effort")),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        run = snapshot["run"]
        return {
            "run_id": run["run_id"],
            "workflow_id": run["workflow_id"],
            "status": run["status"],
            "convo_id": run["convo_id"],
            "snapshot": snapshot,
        }

    @router.get("/workflow-runs")
    async def list_runs(limit: int = 50):
        return {"runs": await store_getter().list_runs(limit=limit)}

    @router.get("/workflow-runs/{run_id}")
    async def get_run(run_id: str):
        snapshot = await store_getter().snapshot(run_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail=f"unknown workflow run: {run_id}")
        return snapshot

    @router.post("/workflow-runs/{run_id}/advance")
    async def advance_run(run_id: str):
        try:
            return await executor_getter().advance(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/workflow-runs/{run_id}/run")
    async def run_to_completion(run_id: str, request: Request):
        body = await _json_body(request, allow_empty=True)
        max_steps = body.get("max_steps") if isinstance(body, dict) else None
        try:
            return await executor_getter().run_to_completion(run_id, max_steps=max_steps)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/workflow-runs/{run_id}/steps/{step_id}/retry")
    async def retry_step(run_id: str, step_id: str):
        try:
            return await executor_getter().retry_step(run_id, step_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return router


async def _json_body(request: Request, *, allow_empty: bool = False) -> dict[str, Any]:
    raw = await request.body()
    if not raw and allow_empty:
        return {}
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Invalid JSON object")
    return body


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
