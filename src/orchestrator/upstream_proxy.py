from __future__ import annotations

import asyncio
import json
from typing import Dict, Optional

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from . import proxy_services as services
from .llm_router import get_router
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
    """Handles HTTP proxying to upstream endpoints."""

    @staticmethod
    async def stream_response(
        request: Request,
        target_url: str,
        body: Dict,
        convo_id: Optional[str],
        extra_headers: Dict = None,
        trace: RequestTrace = None,
    ) -> StreamingResponse:
        """Handle streaming response."""
        timeout = httpx.Timeout(connect=20, read=None, write=20, pool=20)
        headers = HeaderManager.prepare_upstream_headers(request, for_streaming=True)
        body["stream_options"] = {"include_usage": True}

        acc = SSEAccumulator()
        chunks: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        downstream_open = True

        async def emit(chunk: bytes) -> None:
            if downstream_open:
                await chunks.put(chunk)

        async def produce():
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream(
                        "POST", target_url, headers=headers, json=body
                    ) as resp:
                        resp.raise_for_status()

                        start_reasoning_buffer = ""
                        end_reasoning_buffer = ""
                        async for line in resp.aiter_lines():
                            (
                                chunk,
                                start_reasoning_buffer,
                                end_reasoning_buffer,
                                should_continue,
                            ) = await process_stream_line(
                                line, acc, start_reasoning_buffer, end_reasoning_buffer
                            )

                            if chunk:
                                await emit(chunk)

                            if not should_continue:
                                break

            except httpx.HTTPStatusError as e:
                if trace:
                    trace.status_code = e.response.status_code
                    trace.error_class = type(e).__name__
                await emit(
                    create_error_sse_message(
                        "error",
                        status=e.response.status_code,
                        detail=f"HTTP {e.response.status_code}",
                    )
                )
            except Exception as e:
                if trace:
                    trace.status_code = 500
                    trace.error_class = type(e).__name__
                await emit(create_error_sse_message("error", detail=str(e)))
            finally:
                await RequestProcessor.finalize_stream_response(
                    convo_id, acc.text(), trace
                )
                if downstream_open:
                    await chunks.put(None)

        services.track_stream_producer(asyncio.create_task(produce()))

        async def stream():
            nonlocal downstream_open
            try:
                while True:
                    chunk = await chunks.get()
                    if chunk is None:
                        break
                    yield chunk
            finally:
                downstream_open = False

        response_headers = HeaderManager.create_response_headers(
            convo_id=convo_id, for_streaming=True
        )
        if extra_headers:
            response_headers.update(extra_headers)
        if trace:
            trace.apply_headers(response_headers)
            try:
                trace.emit(status_code=200, phase="started")
            except Exception:
                services.logger.exception("Failed to emit request trace")

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers=response_headers
        )

    @staticmethod
    async def non_stream_response(
        request: Request,
        target_url: str,
        body: Dict,
        convo_id: Optional[str],
        extra_headers: Dict = None,
        trace: RequestTrace = None,
    ) -> Response:
        """Handle non-streaming response."""
        headers = HeaderManager.prepare_upstream_headers(request)

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(target_url, headers=headers, json=body)
                resp.raise_for_status()

                # Process response and extract reasoning
                try:
                    resp_json = resp.json()
                    processed_resp = process_non_stream_response(resp_json)

                    # Extract assistant text for history (from processed response)
                    assistant_text = extract_assistant_text(processed_resp)

                    if "model" in processed_resp:
                        services.logger.info(f"Response model: {processed_resp['model']}")

                    # Update response content with processed version
                    resp_content = json.dumps(processed_resp).encode("utf-8")

                except Exception:
                    services.logger.info(f"Response: {resp.text[-500:]}")
                    assistant_text = None
                    resp_content = resp.content

                history_outcome = await RequestProcessor.update_history(
                    convo_id, assistant_text
                )
                if trace:
                    trace.history.update(history_outcome)

                response_headers = HeaderManager.create_response_headers(
                    dict(resp.headers), convo_id
                )
                if extra_headers:
                    response_headers.update(extra_headers)
                if trace:
                    trace.apply_headers(response_headers)
                    trace.emit(status_code=resp.status_code)

                return Response(resp_content, resp.status_code, response_headers)
        except httpx.HTTPStatusError as exc:
            if trace:
                try:
                    trace.emit(status_code=exc.response.status_code, error=exc)
                except Exception:
                    services.logger.exception("Failed to emit request trace")
            raise
        except Exception as exc:
            if trace:
                try:
                    trace.emit(status_code=500, error=exc)
                except Exception:
                    services.logger.exception("Failed to emit request trace")
            raise


##########  Core Proxy Function ##########
async def proxy_request(
    path: str,
    request: Request,
    extra_headers: Dict = None,
    *,
    route_conflict_policy: str = "reject",
) -> Response:
    """Main proxy handler with canonical conversation-state pinning."""
    endpoint_key = RequestProcessor._endpoint_key(path)
    convo_id = RequestProcessor._normalize_convo_id(
        request.headers.get("X-Convo-ID")
    )
    skip_history = RequestProcessor._truthy_header(
        request.headers.get("X-LLMCP-Skip-History")
    )
    requested_reasoning = RequestProcessor._normalize_reasoning_effort(
        request.headers.get("X-Reasoning-Effort")
    )
    allow_route_switch = RequestProcessor._truthy_header(
        request.headers.get("X-Allow-Route-Switch")
    )
    allow_reasoning_switch = RequestProcessor._truthy_header(
        request.headers.get("X-Allow-Reasoning-Switch")
    )
    valid_endpoints = RequestProcessor.configured_endpoint_names()
    state_headers: Dict[str, str] = {}
    effective_reasoning = requested_reasoning

    if convo_id and not skip_history:
        state_update = await services.history_store.update_conversation_state(
            convo_id,
            route_endpoint=endpoint_key,
            reasoning_effort=requested_reasoning,
            valid_route_endpoints=valid_endpoints,
            allow_route_switch=allow_route_switch,
            allow_reasoning_switch=allow_reasoning_switch,
        )
        if state_update.get("conflict"):
            conflicts = state_update.get("conflicts") or {}
            if (
                route_conflict_policy == "use-existing"
                and set(conflicts) == {"route_endpoint"}
            ):
                endpoint_key = str(conflicts["route_endpoint"])
                path = endpoint_key
                state = await services.history_store.get_conversation_state(convo_id)
                state_headers["X-Route-Pinned"] = "true"
            else:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": "Conversation metadata conflict",
                        "conflicts": conflicts,
                    },
                )
        else:
            state = state_update["state"]
            if state_update.get("route_stale"):
                state_headers["X-Route-Pin-Stale"] = "true"
            switched = state_update.get("switched") or {}
            if isinstance(switched, dict):
                if switched.get("route_endpoint"):
                    state_headers["X-Route-Switched"] = "true"
                    state_headers["X-Route-Previous"] = str(
                        switched["route_endpoint"]
                    )
                    state_headers["X-Route-Decision"] = endpoint_key
                if switched.get("reasoning_effort"):
                    state_headers["X-Reasoning-Switched"] = "true"
                    state_headers["X-Reasoning-Previous"] = str(
                        switched["reasoning_effort"]
                    )
                if switched:
                    state_headers["X-Warning"] = services.SWITCH_WARNING_MESSAGE

        effective_reasoning = RequestProcessor._normalize_reasoning_effort(
            state.get("reasoning_effort")
        )
        state_headers.setdefault("X-Route-Pinned", "false")
        state_headers.setdefault("X-Route-Pin-Stale", "false")
        if effective_reasoning:
            state_headers["X-Reasoning-Effort"] = effective_reasoning

    body, rag_headers = await RequestProcessor.prepare_request(
        request,
        effective_reasoning_effort=effective_reasoning,
        persist_history=not skip_history,
        context_mode="none" if skip_history else None,
    )
    is_streaming = body.get("stream", False) or request.query_params.get("stream") in {
        "true",
        "1",
    }
    response_convo_id = None if skip_history else convo_id
    trace = RequestTrace(convo_id=response_convo_id, endpoint=endpoint_key)

    target_url = RequestProcessor.get_endpoint_url(endpoint_key)
    if not target_url:
        try:
            trace.emit(status_code=404)
        except Exception:
            services.logger.exception("Failed to emit request trace")
        raise HTTPException(status_code=404, detail=f"Endpoint not found: {path}")

    endpoint_config = RequestProcessor.get_endpoint_config(endpoint_key)
    slot_headers = await RequestProcessor.apply_slot_affinity(
        body, None if skip_history else convo_id, endpoint_config
    )

    combined_headers = dict(rag_headers)
    combined_headers.update(state_headers)
    if skip_history:
        combined_headers["X-History-Skipped"] = "true"
    if extra_headers:
        combined_headers.update(extra_headers)
    if state_headers.get("X-Route-Pin-Stale") == "true":
        combined_headers["X-Route-Pin-Stale"] = "true"
    combined_headers.update(slot_headers)
    trace.capture_headers(combined_headers)
    trace.capture_search_from_body(body)

    services.logger.info(
        f"Proxying {request.method} {request.url.path} to {target_url} (streaming: {is_streaming})"
    )

    if is_streaming:
        return await ProxyHandler.stream_response(
            request,
            target_url,
            body,
            response_convo_id,
            combined_headers,
            trace,
        )
    return await ProxyHandler.non_stream_response(
        request, target_url, body, response_convo_id, combined_headers, trace
    )
