from __future__ import annotations

import asyncio
import json

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from starlette.datastructures import Headers

from . import proxy_services as services
from .llm_router import has_images
from .request_processor import RequestProcessor
from .trace import RequestTrace
from .utils import (
    HeaderManager,
    SSEAccumulator,
    create_error_sse_message,
    extract_assistant_text,
    process_non_stream_response,
    process_stream_line,
)


class ProxyHandler:
    @staticmethod
    async def stream_response(
        upstream, conversation_id, response_headers, trace, on_ready
    ):
        """Open upstream before sending headers; keep draining after downstream leaves."""
        acc = SSEAccumulator()
        chunks = asyncio.Queue()
        ready = asyncio.get_running_loop().create_future()
        # The caller can leave before an upstream error arrives.
        ready.add_done_callback(
            lambda future: None if future.cancelled() else future.exception()
        )
        downstream_open = True

        async def produce():
            started = False
            try:
                async with upstream as (target, response):
                    await on_ready(target)
                    trace.apply_headers(response_headers)
                    try:
                        trace.emit(status_code=200, phase="started")
                    except OSError:
                        services.logger.exception("Failed to emit request trace")
                    started = True
                    ready.set_result(None)
                    start_buffer = end_buffer = ""
                    async for line in response.aiter_lines():
                        (
                            chunk,
                            start_buffer,
                            end_buffer,
                            keep_reading,
                        ) = await process_stream_line(
                            line, acc, start_buffer, end_buffer
                        )
                        if chunk and downstream_open:
                            await chunks.put(chunk)
                        if not keep_reading:
                            break
            except asyncio.CancelledError:
                if not ready.done():
                    ready.cancel()
                raise
            except Exception as exc:
                if not started:
                    ready.set_exception(exc)
                else:
                    trace.status_code = (
                        exc.response.status_code
                        if isinstance(exc, httpx.HTTPStatusError)
                        else 500
                    )
                    trace.error_class = type(exc).__name__
                    services.logger.exception(
                        "Upstream stream failed: %s", trace.endpoint
                    )
                    detail = (
                        str(exc).strip()
                        or f"{type(exc).__name__}: upstream response stream failed"
                    )
                    if downstream_open:
                        await chunks.put(
                            create_error_sse_message("error", detail=detail)
                        )
            finally:
                if started:
                    await RequestProcessor.finalize_stream_response(
                        conversation_id, acc.text(), trace
                    )
                if downstream_open:
                    await chunks.put(None)

        services.track_stream_producer(asyncio.create_task(produce()))
        try:
            await asyncio.shield(ready)
        except BaseException:
            downstream_open = False
            raise

        async def stream():
            nonlocal downstream_open
            try:
                while (chunk := await chunks.get()) is not None:
                    yield chunk
            finally:
                downstream_open = False

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers=response_headers
        )

    @staticmethod
    async def non_stream_response(response, conversation_id, response_headers, trace):
        try:
            processed = process_non_stream_response(response.json())
            assistant_text = extract_assistant_text(processed)
            content = json.dumps(processed).encode("utf-8")
        except (ValueError, KeyError, TypeError):
            assistant_text, content = None, response.content
        trace.conversation.update(
            await RequestProcessor.update_conversation(conversation_id, assistant_text)
        )
        headers = HeaderManager.create_response_headers(
            dict(response.headers), conversation_id
        )
        headers.update(response_headers)
        trace.apply_headers(headers)
        trace.emit(status_code=response.status_code)
        return Response(content, response.status_code, headers)


async def proxy_request(path: str, request: Request) -> Response:
    """HTTP boundary; internal clients pass parsed data directly to complete_request."""
    try:
        body = await request.json()
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(400, "Invalid JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected a JSON object")
    if request.query_params.get("stream") in {"true", "1"}:
        body["stream"] = True
    return await complete_request(
        body, request.headers, endpoint=RequestProcessor._endpoint_key(path)
    )


async def complete_request(
    body: dict, headers=None, *, endpoint=None, workload_type=None
) -> Response:
    """Shared chat orchestration for HTTP, workflows, and graphs."""
    headers = Headers(headers or {})
    router = services.llm_router
    if endpoint and workload_type:
        raise HTTPException(400, "Specify endpoint or workload_type, not both")
    if workload_type is not None and workload_type not in router.workloads:
        raise HTTPException(400, f"Unknown workload: {workload_type}")
    automatic = not workload_type and endpoint in {None, "", "smart"}
    endpoint = None if automatic else endpoint
    if endpoint and endpoint not in router.endpoints:
        raise HTTPException(404, f"Endpoint not found: {endpoint}")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(400, "No messages provided")
    if not all(isinstance(m, dict) for m in messages):
        raise HTTPException(400, "Invalid messages")
    if automatic and not any(m.get("role") == "user" for m in messages):
        raise HTTPException(400, "No user messages found")

    conversation_id = RequestProcessor._normalize_conversation_id(
        headers.get("X-Conversation-ID")
    )
    skip = RequestProcessor._truthy_header(headers.get("X-LLMCP-Skip-Conversation"))
    allow_switch = RequestProcessor._truthy_header(headers.get("X-Allow-Route-Switch"))
    effort = RequestProcessor._normalize_reasoning_effort(
        headers.get("X-Reasoning-Effort")
    )
    response_id = None if skip else conversation_id
    response_headers = HeaderManager.create_response_headers(
        conversation_id=response_id, for_streaming=bool(body.get("stream"))
    )

    async def control_state(target=None):
        if not response_id:
            return {}
        update = await services.conversation_store.update_conversation_control_state(
            response_id,
            route_endpoint=target,
            reasoning_effort=effort,
            valid_route_endpoints=list(router.endpoints),
            allow_route_switch=allow_switch,
            allow_reasoning_switch=RequestProcessor._truthy_header(
                headers.get("X-Allow-Reasoning-Switch")
            ),
        )
        if update.get("conflict"):
            raise HTTPException(
                409,
                {
                    "message": "Conversation metadata conflict",
                    "conflicts": update["conflicts"],
                },
            )
        response_headers.setdefault("X-Route-Pin-Stale", "false")
        if update.get("route_stale"):
            response_headers["X-Route-Pin-Stale"] = "true"
        for key, previous in (update.get("switched") or {}).items():
            prefix = "X-Route" if key == "route_endpoint" else "X-Reasoning"
            response_headers[f"{prefix}-Switched"] = "true"
            response_headers[f"{prefix}-Previous"] = str(previous)
            response_headers["X-Warning"] = services.SWITCH_WARNING_MESSAGE
        return update["state"]

    state = await control_state(endpoint)
    effort = (
        RequestProcessor._normalize_reasoning_effort(state.get("reasoning_effort"))
        or effort
    )
    pinned = state.get("route_endpoint") if automatic and not allow_switch else None
    if pinned and pinned not in router.workloads.get("vision", []):
        history = await services.conversation_store.get_conversation(response_id) or []
        if has_images(messages) or has_images(history):
            raise HTTPException(
                409,
                "Pinned endpoint is not configured for vision; allow a route switch or select a vision endpoint",
            )
    body, retrieval_headers = await RequestProcessor.prepare_request(
        dict(body),
        headers,
        effective_reasoning_effort=effort,
        persist_conversation=not skip,
        history_mode="none" if skip else None,
    )
    response_headers.update(retrieval_headers)
    if skip:
        response_headers["X-Conversation-Skipped"] = "true"
    if automatic:
        if pinned:
            endpoint = pinned
        else:
            workload_type = await router.classify(body["messages"])
    response_headers["X-Route-Pinned"] = str(bool(pinned)).lower()
    trace = RequestTrace(
        conversation_id=response_id, endpoint=endpoint or workload_type
    )

    async def before_request(target, payload):
        slot_headers = await RequestProcessor.apply_slot_affinity(
            payload, response_id, target
        )
        # A failed candidate must not leave slot metadata on a later candidate.
        for key in list(response_headers):
            if key.startswith("X-Upstream-Slot-"):
                del response_headers[key]
        response_headers.update(slot_headers)

    async def on_ready(target):
        await control_state(target["name"])
        trace.endpoint = target["name"]
        response_headers["X-Route-Decision"] = target["name"]
        response_headers["X-Route-Strategy"] = workload_type or (
            "pinned" if pinned else "explicit"
        )
        for key in ("gpu", "vram", "soc", "cpu", "ram"):
            if target.get(key):
                response_headers[f"X-Route-{key.upper()}"] = str(target[key])
        trace.capture_headers(response_headers)
        trace.capture_search_from_body(body)

    if body.get("stream"):
        body["stream_options"] = {"include_usage": True}
    timeout = (
        httpx.Timeout(connect=20, read=None, write=20, pool=20)
        if body.get("stream")
        else 120
    )
    upstream = router.request(
        body,
        endpoint=endpoint,
        workload_type=workload_type,
        headers=headers,
        timeout=timeout,
        before_request=before_request,
    )
    try:
        if body.get("stream"):
            return await ProxyHandler.stream_response(
                upstream, response_id, response_headers, trace, on_ready
            )
        async with upstream as (target, response):
            await on_ready(target)
            return await ProxyHandler.non_stream_response(
                response, response_id, response_headers, trace
            )
    except (httpx.HTTPError, HTTPException, TimeoutError) as exc:
        status = (
            exc.status_code
            if isinstance(exc, HTTPException)
            else (
                exc.response.status_code
                if isinstance(exc, httpx.HTTPStatusError)
                else 502
            )
        )
        trace.emit(status_code=status, error=exc)
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status, str(exc)) from exc
