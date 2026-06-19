from __future__ import annotations

import json
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from jsonschema import ValidationError

from .executor import GraphExecutor
from .registry import GraphRegistry
from .store import SQLiteGraphRunStore


def create_graph_router(
    *,
    registry_getter: Callable[[], GraphRegistry],
    store_getter: Callable[[], SQLiteGraphRunStore],
    executor_getter: Callable[[], GraphExecutor],
) -> APIRouter:
    router = APIRouter()

    @router.get("/graphs")
    async def list_graphs():
        return {"graphs": [spec.summary_dict() for spec in registry_getter().list()]}

    @router.get("/graphs/{graph_id}")
    async def get_graph(graph_id: str):
        try:
            return registry_getter().get(graph_id).to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/graphs/{graph_id}/runs")
    async def create_run(graph_id: str, request: Request):
        body = await _json_body(request)
        try:
            return await executor_getter().create_run(
                graph_id,
                input=_dict_or_empty(body.get("input")),
                config=_dict_or_empty(body.get("config")),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=exc.message) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/graph-runs")
    async def list_runs(limit: int = 50):
        return {"runs": await store_getter().list_runs(limit=limit)}

    @router.get("/graph-runs/{run_id}")
    async def get_run(run_id: str):
        snapshot = await store_getter().snapshot(run_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail=f"unknown graph run: {run_id}")
        return snapshot

    @router.post("/graph-runs/{run_id}/run")
    async def run_to_completion(run_id: str):
        try:
            return await executor_getter().run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/graph-runs/{run_id}/stream")
    async def stream_run(run_id: str, request: Request):
        body = await _json_body(request, allow_empty=True)
        stream_mode = body.get("stream_mode") if isinstance(body, dict) else None

        async def events():
            async for event in executor_getter().stream(run_id, stream_mode=stream_mode):
                yield _sse_event(event)

        return StreamingResponse(events(), media_type="text/event-stream")

    @router.post("/graph-runs/{run_id}/resume")
    async def resume_run(run_id: str, request: Request):
        body = await _json_body(request, allow_empty=True)
        try:
            return await executor_getter().resume(
                run_id,
                resume=body.get("resume"),
                config=_dict_or_empty(body.get("config")),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

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


def _sse_event(event: dict[str, Any]) -> bytes:
    event_type = str(event.get("type") or "message")
    return (
        f"event: {event_type}\n"
        f"data: {json.dumps(event, default=str)}\n\n"
    ).encode("utf-8")

