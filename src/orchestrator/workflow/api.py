from __future__ import annotations

import json
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..thread_briefing import (
    build_thread_briefing_bundle,
    enrich_workflow_params_with_thread_bundle,
)
from ..conversation_store import ConversationStore
from .executor import WorkflowExecutor
from .registry import WorkflowRegistry
from .store import SQLiteWorkflowStore


def create_workflow_router(
    *,
    registry_getter: Callable[[], WorkflowRegistry],
    store_getter: Callable[[], SQLiteWorkflowStore],
    executor_getter: Callable[[], WorkflowExecutor],
    conversation_store_getter: Callable[[], ConversationStore] | None = None,
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
        params = _dict_or_empty(body.get("params"))
        if "context_mode" in body:
            raise HTTPException(
                status_code=400,
                detail="context_mode is no longer supported; use history_mode",
            )
        history_mode = str(body.get("history_mode") or "").strip().lower()
        if history_mode and history_mode not in {"conversation", "thread", "none"}:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported history mode: {history_mode}",
            )
        if history_mode == "thread":
            if conversation_store_getter is None:
                raise HTTPException(
                    status_code=400,
                    detail="workflow thread briefing is not configured",
                )
            source_conversation_id = _optional_str(body.get("source_conversation_id"))
            if not source_conversation_id:
                raise HTTPException(
                    status_code=400,
                    detail="source_conversation_id is required for workflow thread mode",
                )
            recent_message_count = _optional_int(body.get("recent_message_count")) or 20
            endpoint = _optional_str(body.get("endpoint"))
            if _truthy(body.get("refresh_thread_state")):
                from ..thread_state_refresh import (
                    maybe_refresh_thread_state,
                )

                if not endpoint:
                    raise HTTPException(
                        status_code=400,
                        detail="endpoint is required to refresh thread state",
                    )
                try:
                    await maybe_refresh_thread_state(
                        conversation_store=conversation_store_getter(),
                        llm_client=executor_getter().llm_client,
                        source_conversation_id=source_conversation_id,
                        endpoint=endpoint,
                        reasoning_effort=_optional_str(body.get("reasoning_effort")),
                        force=False,
                    )
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=str(exc)) from exc
            bundle = await build_thread_briefing_bundle(
                conversation_store_getter(),
                source_conversation_id=source_conversation_id,
                recent_message_count=recent_message_count,
            )
            params = enrich_workflow_params_with_thread_bundle(params, bundle)
        try:
            snapshot = await executor_getter().create_run(
                workflow_id,
                params=params,
                conversation_id=_optional_str(body.get("conversation_id")),
                endpoint=_optional_str(body.get("endpoint")),
                reasoning_effort=_optional_str(body.get("reasoning_effort")),
                retrieval_endpoint=_optional_str(body.get("retrieval_endpoint")),
                search_provider=_optional_str(body.get("search_provider")),
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
            "conversation_id": run["conversation_id"],
            "snapshot": snapshot,
        }

    @router.get("/workflow-runs")
    async def list_runs(limit: int = 50):
        return {"runs": await store_getter().list_runs(limit=limit)}

    @router.delete("/workflow-runs")
    async def clear_runs():
        return {"deleted": await store_getter().clear_runs()}

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

    @router.post("/workflow-runs/{run_id}/run-stream")
    async def run_to_completion_stream(run_id: str, request: Request):
        body = await _json_body(request, allow_empty=True)
        max_steps = body.get("max_steps") if isinstance(body, dict) else None
        stream_llm = bool(body.get("stream", True)) if isinstance(body, dict) else True

        async def events():
            async for event in executor_getter().run_to_completion_stream(
                run_id,
                max_steps=max_steps,
                stream_llm=stream_llm,
            ):
                yield _sse_event(event)

        return StreamingResponse(events(), media_type="text/event-stream")

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


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"expected integer: {value!r}") from exc


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _sse_event(event: dict[str, Any]) -> bytes:
    event_type = str(event.get("type") or "message")
    return (
        f"event: {event_type}\n"
        f"data: {json.dumps(event, default=str)}\n\n"
    ).encode("utf-8")
