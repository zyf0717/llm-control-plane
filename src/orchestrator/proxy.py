import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from src.search import (
    EPHEMERAL_WEB_SEARCH_CONTEXT_MARKER,
    SearchArgs,
    build_search_router,
    wrap_search_results,
)

from .config import CONFIG_FILE, load_config
from .history_store import (
    HistoryStore,
    MemoryHistoryStore,
    build_history_store_from_env,
)
from .llm_router import get_router
from .trace import RequestTrace
from .utils import (
    HeaderManager,
    SSEAccumulator,
    create_error_sse_message,
    extract_assistant_text,
    process_non_stream_response,
    process_stream_line,
)

# Configuration
load_dotenv()
config = load_config(CONFIG_FILE)
endpoints = config.get("endpoints", [])
rag_config = config.get("rag", {})
RAG_TOP_K = int(rag_config.get("top_k", 3))
search_service = build_search_router(
    config.get("search", {}),
    planner_headers=HeaderManager.create_auth_headers(),
)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI()

# Global conversation store
history_store: HistoryStore = MemoryHistoryStore()

# Global cache of reachable endpoints
reachable_endpoints: Dict[str, dict] = {}
VALID_REASONING_EFFORTS = {"low", "medium", "high"}


def set_history_store(store: HistoryStore) -> None:
    """Swap the active history store implementation."""
    global history_store
    history_store = store


async def initialize_history_store() -> None:
    """Initialize the configured history store with memory fallback."""
    global history_store

    candidate = build_history_store_from_env()
    try:
        await candidate.initialize()
    except Exception:
        logger.exception(
            "Failed to initialize %s history store; falling back to memory",
            candidate.backend_name,
        )
        fallback = MemoryHistoryStore()
        await fallback.initialize()
        history_store = fallback
    else:
        history_store = candidate

    logger.info("Conversation history backend: %s", history_store.backend_name)


@app.on_event("startup")
async def startup_history_store() -> None:
    await initialize_history_store()


@app.on_event("shutdown")
async def shutdown_history_store() -> None:
    await history_store.close()


class RequestProcessor:
    """Handles request processing, history management, and routing logic."""

    @staticmethod
    def _endpoint_key(path: str) -> str:
        """Extract the configured endpoint name from a proxy path."""
        return path.lstrip("/").split("/")[0] if path else ""

    @staticmethod
    def configured_endpoint_names() -> List[str]:
        """Return currently configured concrete endpoint names."""
        return [
            str(endpoint.get("name") or "").strip()
            for endpoint in endpoints
            if str(endpoint.get("name") or "").strip()
        ]

    @staticmethod
    def get_endpoint_config(path_or_endpoint: str) -> Optional[Dict[str, Any]]:
        """Return endpoint config by path or endpoint name."""
        endpoint_key = RequestProcessor._endpoint_key(path_or_endpoint)
        for endpoint_config in endpoints:
            if endpoint_config.get("name") == endpoint_key:
                return endpoint_config
        return None

    @staticmethod
    def get_endpoint_url(path: str) -> Optional[str]:
        """Get target endpoint URL based on path and streaming preference."""
        endpoint_config = RequestProcessor.get_endpoint_config(path)
        if endpoint_config:
            base_url = endpoint_config.get("url")
            if base_url:
                return f"{base_url}/v1/chat/completions"

        logger.warning(f"No endpoint found for path: {path}")
        return None

    @staticmethod
    def _normalize_convo_id(raw_convo_id: Optional[str]) -> Optional[str]:
        convo_id = str(raw_convo_id or "").strip()
        return convo_id or None

    @staticmethod
    def _normalize_reasoning_effort(raw_effort: Optional[str]) -> Optional[str]:
        effort = str(raw_effort or "").strip().lower()
        return effort if effort in VALID_REASONING_EFFORTS else None

    @staticmethod
    def _normalize_rag_endpoint(rag_endpoint: Optional[str]) -> Optional[str]:
        """Normalize a configured RAG endpoint into the context retrieval URL."""
        if not rag_endpoint:
            return None

        normalized = rag_endpoint.strip().rstrip("/")
        normalized = rag_endpoint.strip()

        # Handle UI display strings like "Name (http://url)"
        if " " in normalized:
            import re

            match = re.search(r"\((https?://[^)]+)\)", normalized)
            if match:
                normalized = match.group(1)
            else:
                normalized = normalized.split()[0]

        normalized = normalized.rstrip("/")
        if not normalized:
            return None
        if not normalized.startswith(("http://", "https://")):
            normalized = f"http://{normalized}"

        if normalized.endswith("/api/retrieve/context"):
            return normalized
        if normalized.endswith("/api/retrieve"):
            return f"{normalized}/context"
        if normalized.endswith("/context"):
            return normalized
        return f"{normalized}/api/retrieve/context"

    @staticmethod
    def _latest_user_message(messages: List[Dict]) -> Optional[str]:
        """Get the latest user message content from the message list."""
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
        return None

    @staticmethod
    def _rag_insertion_index(messages: List[Dict]) -> int:
        """Insert RAG context immediately before the latest user turn."""
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "user":
                return index
        return len(messages)

    @staticmethod
    def _is_ephemeral_search_message(message: Dict) -> bool:
        """Identify turn-local search wrappers that must not persist or replay."""
        if not isinstance(message, dict):
            return False

        content = message.get("content")
        if not isinstance(content, str):
            return False

        return content.strip().startswith(EPHEMERAL_WEB_SEARCH_CONTEXT_MARKER)

    @staticmethod
    def _is_ephemeral_rag_message(message: Dict) -> bool:
        """Identify turn-local RAG wrappers that must not be replay anchors."""
        if not isinstance(message, dict):
            return False

        content = message.get("content")
        if not isinstance(content, str):
            return False

        stripped = content.strip()
        return stripped.startswith("Retrieved reference excerpts:") or stripped.startswith(
            "[Retrieved reference excerpts]"
        )

    @staticmethod
    def _is_ephemeral_history_message(message: Dict) -> bool:
        return RequestProcessor._is_ephemeral_search_message(
            message
        ) or RequestProcessor._is_ephemeral_rag_message(message)

    @staticmethod
    def _filter_ephemeral_search_messages(messages: List[Dict]) -> List[Dict]:
        """Drop synthetic context from durable history/replay boundaries."""
        return [
            message
            for message in messages
            if not RequestProcessor._is_ephemeral_history_message(message)
        ]

    @staticmethod
    def _is_full_history_replay(stored: List[Dict], incoming: List[Dict]) -> bool:
        """Detect clients replaying the full server-persisted history prefix."""
        if not stored or len(incoming) < len(stored):
            return False
        return incoming[: len(stored)] == stored

    @staticmethod
    def _merge_ephemeral_search_context(messages: List[Dict]) -> List[Dict]:
        """Attach ephemeral search context to the next user turn for upstream chat."""
        merged: List[Dict] = []
        pending_context: List[str] = []

        for message in messages:
            if RequestProcessor._is_ephemeral_search_message(message):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    pending_context.append(content.strip())
                continue

            if pending_context and message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    message = dict(message)
                    message["content"] = "\n\n".join(
                        [*pending_context, f"Current user question:\n{content}"]
                    )
                    pending_context = []

            merged.append(message)

        return merged

    @staticmethod
    async def _fetch_rag_message(
        messages: List[Dict], rag_endpoint: Optional[str]
    ) -> tuple[Optional[str], Dict[str, str]]:
        """Fetch a pre-grounded user message from the context service."""
        normalized_endpoint = RequestProcessor._normalize_rag_endpoint(rag_endpoint)
        if not normalized_endpoint:
            return None, {}

        latest_user_message = RequestProcessor._latest_user_message(messages)
        if not latest_user_message:
            return None, {
                "X-RAG-Endpoint": normalized_endpoint,
                "X-RAG-Injected": "false",
                "X-RAG-Reason": "no-user-message",
            }

        request_headers = HeaderManager.create_auth_headers()
        request_headers["Content-Type"] = "application/json"
        request_payload = {
            "query": latest_user_message,
            "limit": RAG_TOP_K,
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    normalized_endpoint, json=request_payload, headers=request_headers
                )
                response.raise_for_status()
                search_response = response.json()
        except Exception as exc:
            logger.warning(
                "RAG context retrieval failed for %s: %s", normalized_endpoint, exc
            )
            return None, {
                "X-RAG-Endpoint": normalized_endpoint,
                "X-RAG-Injected": "false",
                "X-RAG-Reason": "request-failed",
            }

        context_blocks = search_response.get("context_blocks", [])
        if not isinstance(context_blocks, list):
            context_blocks = []

        grounded_user_message = search_response.get("grounded_user_message")
        if (
            not isinstance(grounded_user_message, str)
            or not grounded_user_message.strip()
        ):
            logger.info(
                "RAG context returned no grounded user message via %s",
                normalized_endpoint,
            )
            return None, {
                "X-RAG-Endpoint": normalized_endpoint,
                "X-RAG-Injected": "false",
                "X-RAG-Reason": "empty-grounded-user-message",
            }

        rag_headers = {
            "X-RAG-Endpoint": normalized_endpoint,
            "X-RAG-Hits": str(len(context_blocks)),
            "X-RAG-Injected": "true",
        }
        if search_response.get("mode") is not None:
            rag_headers["X-RAG-Mode"] = str(search_response["mode"])
        if search_response.get("truncated") is not None:
            rag_headers["X-RAG-Truncated"] = str(
                bool(search_response["truncated"])
            ).lower()

        logger.info(
            "RAG context retrieved %d blocks via %s (mode=%s truncated=%s)",
            len(context_blocks),
            normalized_endpoint,
            str(search_response.get("mode") or "unknown"),
            bool(search_response.get("truncated", False)),
        )
        return grounded_user_message.strip(), rag_headers

    @staticmethod
    async def prepare_request(
        request: Request, effective_reasoning_effort: Optional[str] = None
    ) -> tuple[Dict, Dict[str, str]]:
        """Parse and enrich request with conversation history and reasoning."""
        try:
            raw_body = await request.body()
            body = json.loads(raw_body) if raw_body else {}
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON")

        if not isinstance(body, dict):
            return body, {}

        convo_id = RequestProcessor._normalize_convo_id(
            request.headers.get("X-Convo-ID")
        )
        rag_endpoint = request.headers.get("X-RAG-Endpoint")
        reasoning_effort = (
            effective_reasoning_effort
            or RequestProcessor._normalize_reasoning_effort(
                request.headers.get("X-Reasoning-Effort")
            )
        )

        incoming_messages = body.get("messages", [])
        if not isinstance(incoming_messages, list):
            incoming_messages = []

        stored_messages = []
        if convo_id:
            stored_messages = RequestProcessor._filter_ephemeral_search_messages(
                await history_store.get_conversation(convo_id) or []
            )
            durable_incoming_messages = (
                RequestProcessor._filter_ephemeral_search_messages(incoming_messages)
            )
            if RequestProcessor._is_full_history_replay(
                stored_messages, durable_incoming_messages
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Full-history payload replay is not allowed when X-Convo-ID is present",
                )
            if durable_incoming_messages:
                await history_store.append_messages(convo_id, durable_incoming_messages)

        messages = [*stored_messages, *incoming_messages]

        if messages:
            if (
                messages
                and messages[0].get("role") == "system"
                and "Reasoning:" in messages[0].get("content", "")
            ):
                messages.pop(0)

            if reasoning_effort:
                reasoning_msg = {
                    "role": "system",
                    "content": f"Reasoning: {reasoning_effort}",
                }
                messages.insert(0, reasoning_msg)
                logger.info(f"Applied reasoning: {reasoning_effort}")

        rag_user_content, rag_headers = await RequestProcessor._fetch_rag_message(
            messages, rag_endpoint
        )
        if rag_user_content:
            latest_user_index = RequestProcessor._rag_insertion_index(messages)
            rewritten_user_message = dict(messages[latest_user_index])
            rewritten_user_message["content"] = rag_user_content
            messages[latest_user_index] = rewritten_user_message

        messages = RequestProcessor._merge_ephemeral_search_context(messages)
        body["messages"] = messages
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
            rag_headers["X-Reasoning-Effort"] = reasoning_effort

        return body, rag_headers

    @staticmethod
    async def apply_slot_affinity(
        body: Dict, convo_id: Optional[str], endpoint_config: Optional[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Best-effort llama.cpp slot affinity for configured endpoints."""
        if (
            not convo_id
            or not endpoint_config
            or not endpoint_config.get("slot_affinity")
        ):
            return {}

        endpoint = str(endpoint_config.get("name") or "").strip()
        base_url = str(endpoint_config.get("url") or "").strip().rstrip("/")
        if not endpoint or not base_url:
            return {"X-Upstream-Slot-Status": "disabled"}

        try:
            state = await history_store.get_conversation_state(convo_id)
            slots = state.get("slots") if isinstance(state, dict) else {}
            slot_id = (slots or {}).get(endpoint) if isinstance(slots, dict) else None
            if slot_id is None:
                slot_id = await RequestProcessor._probe_llama_slot(base_url)
                if slot_id is None:
                    return {"X-Upstream-Slot-Status": "unavailable"}
                await history_store.set_conversation_slot(
                    convo_id, endpoint, int(slot_id)
                )

            body["id_slot"] = int(slot_id)
            body["cache_prompt"] = True
            return {
                "X-Upstream-Slot-ID": str(slot_id),
                "X-Upstream-Slot-Status": "affinity-applied",
            }
        except Exception as exc:
            logger.info("Slot affinity skipped for %s/%s: %s", endpoint, convo_id, exc)
            return {"X-Upstream-Slot-Status": "skipped"}

    @staticmethod
    async def _probe_llama_slot(base_url: str) -> Optional[int]:
        headers = HeaderManager.create_auth_headers()
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(
                    f"{base_url}/slots?fail_on_no_slot=1", headers=headers
                )
                response.raise_for_status()
                slots = response.json()
        except Exception:
            return None

        if not isinstance(slots, list) or not slots:
            return None

        ordered_slots = sorted(
            [slot for slot in slots if isinstance(slot, dict) and "id" in slot],
            key=lambda slot: (bool(slot.get("is_processing")), int(slot.get("id", 0))),
        )
        if not ordered_slots:
            return None
        try:
            return int(ordered_slots[0]["id"])
        except (TypeError, ValueError):
            return None

    @staticmethod
    async def update_history(convo_id: Optional[str], assistant_text: str) -> None:
        """Update conversation history with assistant response."""
        if assistant_text and convo_id:
            await history_store.append_messages(
                convo_id, [{"role": "assistant", "content": assistant_text}]
            )


class ProxyHandler:
    """Handles HTTP proxying to upstream endpoints."""

    @staticmethod
    async def stream_response(
        request: Request,
        target_url: str,
        body: Dict,
        convo_id: str,
        extra_headers: Dict = None,
        trace: RequestTrace = None,
    ) -> StreamingResponse:
        """Handle streaming response."""
        timeout = httpx.Timeout(connect=20, read=None, write=20, pool=20)
        headers = HeaderManager.prepare_upstream_headers(request, for_streaming=True)
        body["stream_options"] = {"include_usage": True}

        acc = SSEAccumulator()

        async def stream():
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
                                yield chunk

                            if not should_continue:
                                break

            except httpx.HTTPStatusError as e:
                yield create_error_sse_message(
                    "error",
                    status=e.response.status_code,
                    detail=f"HTTP {e.response.status_code}",
                )
            except Exception as e:
                yield create_error_sse_message("error", detail=str(e))
            finally:
                try:
                    await RequestProcessor.update_history(convo_id, acc.text())
                except Exception:
                    pass

        response_headers = HeaderManager.create_response_headers(
            convo_id=convo_id, for_streaming=True
        )
        if extra_headers:
            response_headers.update(extra_headers)
        if trace:
            trace.apply_headers(response_headers)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers=response_headers
        )

    @staticmethod
    async def non_stream_response(
        request: Request,
        target_url: str,
        body: Dict,
        convo_id: str,
        extra_headers: Dict = None,
        trace: RequestTrace = None,
    ) -> Response:
        """Handle non-streaming response."""
        headers = HeaderManager.prepare_upstream_headers(request)

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
                    logger.info(f"Response model: {processed_resp['model']}")

                # Update response content with processed version
                resp_content = json.dumps(processed_resp).encode("utf-8")

            except Exception:
                logger.info(f"Response: {resp.text[-500:]}")
                assistant_text = None
                resp_content = resp.content

            await RequestProcessor.update_history(convo_id, assistant_text)

            response_headers = HeaderManager.create_response_headers(
                dict(resp.headers), convo_id
            )
            if extra_headers:
                response_headers.update(extra_headers)
            if trace:
                trace.mark_elapsed()
                trace.apply_headers(response_headers)

            return Response(resp_content, resp.status_code, response_headers)


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
    requested_reasoning = RequestProcessor._normalize_reasoning_effort(
        request.headers.get("X-Reasoning-Effort")
    )
    valid_endpoints = RequestProcessor.configured_endpoint_names()
    state_headers: Dict[str, str] = {}
    effective_reasoning = requested_reasoning

    if convo_id:
        state_update = await history_store.update_conversation_state(
            convo_id,
            route_endpoint=endpoint_key,
            reasoning_effort=requested_reasoning,
            valid_route_endpoints=valid_endpoints,
        )
        if state_update.get("conflict"):
            conflicts = state_update.get("conflicts") or {}
            if (
                route_conflict_policy == "use-existing"
                and set(conflicts) == {"route_endpoint"}
            ):
                endpoint_key = str(conflicts["route_endpoint"])
                path = endpoint_key
                state = await history_store.get_conversation_state(convo_id)
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

        effective_reasoning = RequestProcessor._normalize_reasoning_effort(
            state.get("reasoning_effort")
        )
        state_headers.setdefault("X-Route-Pinned", "false")
        state_headers.setdefault("X-Route-Pin-Stale", "false")
        if effective_reasoning:
            state_headers["X-Reasoning-Effort"] = effective_reasoning

    body, rag_headers = await RequestProcessor.prepare_request(
        request, effective_reasoning_effort=effective_reasoning
    )
    is_streaming = body.get("stream", False) or request.query_params.get("stream") in {
        "true",
        "1",
    }
    trace = RequestTrace(convo_id=convo_id, endpoint=endpoint_key)

    target_url = RequestProcessor.get_endpoint_url(endpoint_key)
    if not target_url:
        raise HTTPException(status_code=404, detail=f"Endpoint not found: {path}")

    endpoint_config = RequestProcessor.get_endpoint_config(endpoint_key)
    slot_headers = await RequestProcessor.apply_slot_affinity(
        body, convo_id, endpoint_config
    )

    combined_headers = dict(rag_headers)
    combined_headers.update(state_headers)
    if extra_headers:
        combined_headers.update(extra_headers)
    if state_headers.get("X-Route-Pin-Stale") == "true":
        combined_headers["X-Route-Pin-Stale"] = "true"
    combined_headers.update(slot_headers)
    trace.capture_headers(combined_headers)

    logger.info(
        f"Proxying {request.method} {request.url.path} to {target_url} (streaming: {is_streaming})"
    )

    if is_streaming:
        return await ProxyHandler.stream_response(
            request, target_url, body, convo_id, combined_headers, trace
        )
    return await ProxyHandler.non_stream_response(
        request, target_url, body, convo_id, combined_headers, trace
    )


##########  API Endpoints ##########
@app.post("/")
async def root_chat(request: Request):
    """Route to smart routing endpoint."""
    logger.info("Routing root request to smart routing")
    return await smart_route(request)


@app.post("/smart")
@app.post("/smart/{subpath:path}")
async def smart_route(request: Request, subpath: str = ""):
    """Smart routing based on content analysis."""
    try:
        body = await request.json()
        messages = body.get("messages", [])

        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user messages found")

        convo_id = RequestProcessor._normalize_convo_id(
            request.headers.get("X-Convo-ID")
        )
        valid_endpoints = RequestProcessor.configured_endpoint_names()
        stale_headers: Dict[str, str] = {}
        if convo_id:
            state = await history_store.get_conversation_state(convo_id)
            pinned_endpoint = str(state.get("route_endpoint") or "").strip()
            if pinned_endpoint and pinned_endpoint in valid_endpoints:
                routing_headers = {
                    "X-Route-Decision": pinned_endpoint,
                    "X-Route-Pinned": "true",
                    "X-Route-Pin-Stale": "false",
                }
                return await proxy_request(
                    pinned_endpoint,
                    request,
                    routing_headers,
                    route_conflict_policy="use-existing",
                )
            if pinned_endpoint and pinned_endpoint not in valid_endpoints:
                await history_store.update_conversation_state(
                    convo_id,
                    valid_route_endpoints=valid_endpoints,
                    clear_route=True,
                )
                stale_headers["X-Route-Pin-Stale"] = "true"

        latest_message = user_messages[-1].get("content", "")

        router = get_router()
        decision = await router.route_request(latest_message, reachable_endpoints)
        endpoint_config = router.get_endpoint_by_name(decision.endpoint)

        routing_headers = {
            "X-Route-Decision": decision.endpoint,
            "X-Route-Confidence": str(decision.confidence),
            "X-Route-Reason": decision.reason,
            "X-Route-Strategy": decision.workload_type.value,
            "X-Route-Pinned": "false",
            "X-Route-Pin-Stale": stale_headers.get("X-Route-Pin-Stale", "false"),
        }

        if endpoint_config:
            for attr in ["gpu", "vram", "soc", "cpu", "ram"]:
                value = getattr(endpoint_config, attr, None)
                if value:
                    routing_headers[f"X-Route-{attr.upper()}"] = value

        logger.info(
            f"Smart routing: {decision.endpoint} (confidence: {decision.confidence:.2f}) - {decision.reason}"
        )

        return await proxy_request(
            decision.endpoint,
            request,
            routing_headers,
            route_conflict_policy="use-existing",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Smart routing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Smart routing error: {str(e)}")


@app.post("/conversations/retrieve")
async def retrieve_conversation(request: Request):
    """Retrieve conversation history."""
    try:
        body = await request.json()
        convo_id = body.get("convo_id")

        if not convo_id:
            raise HTTPException(status_code=400, detail="Missing convo_id")

        conversation = await history_store.get_conversation(convo_id)
        if conversation is None:
            raise HTTPException(
                status_code=404, detail=f"Conversation '{convo_id}' not found"
            )

        return RequestProcessor._filter_ephemeral_search_messages(conversation)

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving conversation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/conversations")
async def list_conversations():
    """List conversation metadata sorted by most recent activity."""
    try:
        return await history_store.list_conversations()
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/models")
async def list_models():
    """List all available models with metadata."""
    models = []

    async def fetch_endpoint_models(endpoint_config):
        """Fetch models from a single endpoint."""
        name = endpoint_config.get("name", "unknown")
        url = endpoint_config.get("url")

        if not url:
            logger.warning(f"No URL for endpoint: {name}")
            return []

        # Special handling for Auto endpoint
        if name == "Auto":
            return [
                {
                    "id": "auto-router",
                    "object": "model",
                    "created": int(datetime.now().timestamp()),
                    "owned_by": "llm-control-plane",
                    "endpoint": name,
                    "endpoint_url": url,
                    "description": "Intelligent routing to best available endpoint",
                }
            ]

        # Fetch models from endpoint
        try:
            headers = HeaderManager.create_auth_headers()

            async with httpx.AsyncClient(timeout=10.0) as client:
                models_urls = [f"{url}/v1/models"]

                resp = None
                for models_url in models_urls:
                    try:
                        resp = await client.get(models_url, headers=headers)
                        resp.raise_for_status()
                        logger.debug(f"Successfully fetched models from {models_url}")
                        break
                    except httpx.HTTPError as e:
                        logger.debug(f"Failed to fetch from {models_url}: {e}")
                        continue

                if resp is None:
                    logger.warning(
                        f"Failed to fetch models from {name} using all endpoints"
                    )
                    return []

                data = resp.json()
                remote_models = data.get(
                    "data", data.get("models", data if isinstance(data, list) else [])
                )

                endpoint_models = []
                for model in remote_models:
                    if isinstance(model, dict):
                        enhanced_model = {
                            "id": model.get("id", f"unknown-{name}"),
                            "object": "model",
                            "created": model.get(
                                "created", int(datetime.now().timestamp())
                            ),
                            "owned_by": model.get("owned_by", name),
                            "endpoint": name,
                            "endpoint_url": url,
                        }

                        # Add hardware specs
                        for hw in ["gpu", "vram", "soc", "cpu", "ram"]:
                            if hw in endpoint_config:
                                enhanced_model[hw] = endpoint_config[hw]

                        # Preserve original fields
                        for k, v in model.items():
                            if k not in enhanced_model:
                                enhanced_model[k] = v

                        endpoint_models.append(enhanced_model)

                return endpoint_models

        except httpx.HTTPError as e:
            logger.warning(f"Failed to fetch models from {name}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching models from {name}: {e}")
            return []

    # Fire off all requests concurrently
    tasks = [fetch_endpoint_models(endpoint_config) for endpoint_config in endpoints]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect all models from successful requests
    for result in results:
        if isinstance(result, list):
            models.extend(result)
        elif isinstance(result, Exception):
            logger.error(f"Endpoint fetch failed: {result}")

    global reachable_endpoints
    reachable_endpoints = list(
        set(model.get("endpoint") for model in models if model.get("endpoint"))
    )
    return {"object": "list", "data": models}


@app.post("/search/web")
async def search_web(request: Request):
    """Return normalized candidate links from configured search providers."""
    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Invalid JSON object")

    try:
        args = SearchArgs(
            query=str(body.get("query", "")).strip(),
            count=body.get("count"),
            provider=str(body.get("provider", "auto") or "auto"),
            context=body.get("context"),
            language=body.get("language"),
            region=body.get("region"),
            safe_search=body.get("safeSearch") or body.get("safe_search"),
            freshness=body.get("freshness"),
        )
        response = await search_service.search(args)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        status_code = 503 if "disabled" in str(exc).lower() else 500
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    payload = response.to_dict()
    payload["wrapped_results"] = wrap_search_results(response)
    return payload


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def custom_endpoints(path: str, request: Request):
    """Handle all other endpoints."""
    return await proxy_request(path, request)
