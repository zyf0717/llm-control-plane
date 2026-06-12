import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from src.logging_config import configure_logging
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
from .workflow import (
    SQLiteWorkflowStore,
    WorkflowExecutor,
    WorkflowRegistry,
    build_workflow_store_from_env,
    create_workflow_router,
)

# Configuration
load_dotenv()
configure_logging()
config = load_config(CONFIG_FILE)
endpoints = config.get("endpoints", [])
rag_config = config.get("rag", {})
RAG_TOP_K = int(rag_config.get("top_k", 3))
search_service = build_search_router(
    config.get("search", {}),
    planner_headers=HeaderManager.create_auth_headers(),
)

logger = logging.getLogger(__name__)

# Global conversation store
history_store: HistoryStore = MemoryHistoryStore()
workflow_registry = WorkflowRegistry()
workflow_store: SQLiteWorkflowStore = build_workflow_store_from_env()
workflow_executor: Optional[WorkflowExecutor] = None

# Global cache of reachable endpoints
reachable_endpoints: Dict[str, dict] = {}
VALID_REASONING_EFFORTS = {"low", "medium", "high"}
history_finalization_tasks: set[asyncio.Task] = set()
stream_producer_tasks: set[asyncio.Task] = set()
SWITCH_WARNING_MESSAGE = (
    "Conversation endpoint/reasoning changed; full history was replayed to the "
    "selected endpoint."
)


def _track_history_finalization(task: asyncio.Task) -> asyncio.Task:
    history_finalization_tasks.add(task)

    def discard_done(done_task: asyncio.Task) -> None:
        history_finalization_tasks.discard(done_task)
        if done_task.cancelled():
            return
        exc = done_task.exception()
        if exc:
            logger.error(
                "Background history finalization failed",
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    task.add_done_callback(discard_done)
    return task


def _track_stream_producer(task: asyncio.Task) -> asyncio.Task:
    stream_producer_tasks.add(task)

    def discard_done(done_task: asyncio.Task) -> None:
        stream_producer_tasks.discard(done_task)
        if done_task.cancelled():
            return
        exc = done_task.exception()
        if exc:
            logger.error(
                "Background stream producer failed",
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    task.add_done_callback(discard_done)
    return task


async def _drain_history_finalizations(timeout: float = 5.0) -> None:
    if not history_finalization_tasks:
        return
    try:
        await asyncio.wait_for(
            asyncio.gather(*history_finalization_tasks, return_exceptions=True),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Timed out waiting for %d history finalization task(s)",
            len(history_finalization_tasks),
        )


async def _drain_stream_producers(timeout: float = 5.0) -> None:
    if not stream_producer_tasks:
        return
    try:
        await asyncio.wait_for(
            asyncio.gather(*stream_producer_tasks, return_exceptions=True),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Timed out waiting for %d stream producer task(s)",
            len(stream_producer_tasks),
        )


def set_history_store(store: HistoryStore) -> None:
    """Swap the active history store implementation."""
    global history_store
    history_store = store


def set_workflow_components(
    *,
    registry: Optional[WorkflowRegistry] = None,
    store: Optional[SQLiteWorkflowStore] = None,
    executor: Optional[WorkflowExecutor] = None,
) -> None:
    """Swap workflow components for tests or alternate runtime wiring."""
    global workflow_registry, workflow_store, workflow_executor
    if registry is not None:
        workflow_registry = registry
    if store is not None:
        workflow_store = store
    if executor is not None:
        workflow_executor = executor


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


async def startup_history_store() -> None:
    await initialize_history_store()


async def shutdown_history_store() -> None:
    await _drain_stream_producers()
    await _drain_history_finalizations()
    await history_store.close()


async def initialize_workflow_components() -> None:
    """Initialize workflow registry, store, and executor."""
    global workflow_registry, workflow_store, workflow_executor
    workflow_registry.load()
    workflow_store = build_workflow_store_from_env()
    await workflow_store.initialize()
    workflow_executor = WorkflowExecutor(
        workflow_registry,
        workflow_store,
        ProxyWorkflowLLMClient(),
        ProxyWorkflowSearchClient(),
    )


async def startup_workflow_components() -> None:
    await initialize_workflow_components()


async def shutdown_workflow_components() -> None:
    await workflow_store.close()


def get_workflow_registry() -> WorkflowRegistry:
    return workflow_registry


def get_workflow_store() -> SQLiteWorkflowStore:
    return workflow_store


def get_workflow_executor() -> WorkflowExecutor:
    if workflow_executor is None:
        raise RuntimeError("Workflow executor is not initialized")
    return workflow_executor


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await startup_history_store()
    await startup_workflow_components()
    try:
        yield
    finally:
        await shutdown_workflow_components()
        await shutdown_history_store()


# FastAPI app
app = FastAPI(lifespan=lifespan)


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
    def _truthy_header(raw_value: Optional[str]) -> bool:
        return str(raw_value or "").strip().lower() in {"1", "true", "yes", "on"}

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
                insert_at = (
                    1
                    if messages
                    and messages[0].get("role") == "system"
                    and "Reasoning:" not in messages[0].get("content", "")
                    else 0
                )
                messages.insert(insert_at, reasoning_msg)
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
    async def update_history(
        convo_id: Optional[str], assistant_text: Optional[str]
    ) -> Dict[str, Any]:
        """Persist assistant response text and report the durable outcome."""
        text = assistant_text or ""
        outcome: Dict[str, Any] = {
            "assistant_chars": len(text),
            "assistant_persisted": False,
        }
        if not convo_id:
            outcome["assistant_skip_reason"] = "no-convo-id"
            return outcome
        if not text:
            outcome["assistant_skip_reason"] = "empty-assistant"
            return outcome

        await history_store.append_messages(
            convo_id, [{"role": "assistant", "content": text}]
        )
        outcome["assistant_persisted"] = True
        return outcome

    @staticmethod
    async def finalize_stream_history(
        convo_id: Optional[str],
        assistant_text: Optional[str],
        trace: Optional[RequestTrace],
    ) -> None:
        """Persist streamed assistant text and emit completion trace metadata."""
        try:
            history_outcome = await RequestProcessor.update_history(
                convo_id, assistant_text
            )
            if trace:
                trace.history.update(history_outcome)
        except Exception as exc:
            logger.exception(
                "Failed to persist assistant response for convo_id=%s",
                convo_id,
            )
            if trace:
                trace.error_class = trace.error_class or type(exc).__name__
                trace.history.update(
                    {
                        "assistant_chars": len(assistant_text or ""),
                        "assistant_persisted": False,
                        "assistant_error": type(exc).__name__,
                    }
                )

        if trace:
            try:
                phase = "failed" if trace.error_class else "completed"
                trace.emit(
                    status_code=trace.status_code or 200,
                    phase=phase,
                )
            except Exception:
                logger.exception("Failed to emit request trace")

    @staticmethod
    async def finalize_stream_response(
        convo_id: Optional[str],
        assistant_text: Optional[str],
        trace: Optional[RequestTrace],
    ) -> None:
        """Run stream finalization in a task that survives caller cancellation."""
        task = _track_history_finalization(
            asyncio.create_task(
                RequestProcessor.finalize_stream_history(
                    convo_id, assistant_text, trace
                )
            )
        )
        await asyncio.shield(task)


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

        _track_stream_producer(asyncio.create_task(produce()))

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
                logger.exception("Failed to emit request trace")

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
                        logger.info(f"Response model: {processed_resp['model']}")

                    # Update response content with processed version
                    resp_content = json.dumps(processed_resp).encode("utf-8")

                except Exception:
                    logger.info(f"Response: {resp.text[-500:]}")
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
                    logger.exception("Failed to emit request trace")
            raise
        except Exception as exc:
            if trace:
                try:
                    trace.emit(status_code=500, error=exc)
                except Exception:
                    logger.exception("Failed to emit request trace")
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

    if convo_id:
        state_update = await history_store.update_conversation_state(
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
                    state_headers["X-Warning"] = SWITCH_WARNING_MESSAGE

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
        try:
            trace.emit(status_code=404)
        except Exception:
            logger.exception("Failed to emit request trace")
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
    trace.capture_search_from_body(body)

    logger.info(
        f"Proxying {request.method} {request.url.path} to {target_url} (streaming: {is_streaming})"
    )

    if is_streaming:
        return await ProxyHandler.stream_response(
            request,
            target_url,
            body,
            convo_id,
            combined_headers,
            trace,
        )
    return await ProxyHandler.non_stream_response(
        request, target_url, body, convo_id, combined_headers, trace
    )


class _WorkflowRequest:
    def __init__(self, *, path: str, body: Dict[str, Any], headers: Dict[str, str]):
        self.method = "POST"
        self.headers = headers
        self.query_params: Dict[str, str] = {}
        self.url = type("URL", (), {"path": path})()
        self._body = json.dumps(body).encode("utf-8")
        self._json = body

    async def body(self) -> bytes:
        return self._body

    async def json(self) -> Dict[str, Any]:
        return self._json


class ProxyWorkflowLLMClient:
    async def complete(
        self,
        *,
        endpoint: str,
        prompt: str,
        convo_id: str,
        reasoning_effort: Optional[str] = None,
        rag_endpoint: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        endpoint_key = str(endpoint or "smart").strip() or "smart"
        payload: Dict[str, Any] = {
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
        }
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)

        headers = {
            "Content-Type": "application/json",
            "X-Convo-ID": convo_id,
        }
        if reasoning_effort:
            headers["X-Reasoning-Effort"] = reasoning_effort
        if rag_endpoint:
            headers["X-RAG-Endpoint"] = rag_endpoint
        if endpoint_key != "smart":
            headers["X-Allow-Route-Switch"] = "true"

        request = _WorkflowRequest(
            path=f"/{endpoint_key}",
            body=payload,
            headers=headers,
        )
        response = await (
            smart_route(request) if endpoint_key == "smart" else proxy_request(endpoint_key, request)
        )
        response_body = getattr(response, "body", b"") or b""
        try:
            response_json = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            response_json = {}

        text = extract_assistant_text(response_json) or ""
        response_headers = dict(getattr(response, "headers", {}) or {})
        metadata = {
            "endpoint": endpoint_key,
            "status_code": getattr(response, "status_code", None),
        }
        for header_name, header_value in response_headers.items():
            lower_name = str(header_name).lower()
            if lower_name == "x-trace-id":
                metadata["trace_id"] = str(header_value)
            elif lower_name.startswith("x-route-"):
                metadata.setdefault("routing", {})[
                    lower_name.replace("x-route-", "").replace("-", "_")
                ] = str(header_value)
            elif lower_name.startswith("x-rag-"):
                metadata.setdefault("rag", {})[
                    lower_name.replace("x-rag-", "").replace("-", "_")
                ] = str(header_value)

        for key in ("usage", "stats", "model_info", "runtime", "timings"):
            if isinstance(response_json.get(key), dict):
                metadata[key] = response_json[key]

        return {"text": text, "metadata": metadata}


class ProxyWorkflowSearchClient:
    async def search(
        self,
        *,
        query: str,
        provider: Optional[str] = None,
        count: int = 5,
        use_planner: bool = True,
    ) -> Dict[str, Any]:
        response = await search_service.search(
            SearchArgs(
                query=query,
                provider=provider or "auto",
                count=count,
                use_planner=use_planner,
            )
        )
        payload = response.to_dict()
        payload["wrapped_results"] = wrap_search_results(response)
        return payload


app.include_router(
    create_workflow_router(
        registry_getter=get_workflow_registry,
        store_getter=get_workflow_store,
        executor_getter=get_workflow_executor,
    )
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


@app.post("/conversations/state")
async def retrieve_conversation_state(request: Request):
    """Retrieve route/reasoning/slot state for a conversation."""
    try:
        body = await request.json()
        convo_id = body.get("convo_id")

        if not convo_id:
            raise HTTPException(status_code=400, detail="Missing convo_id")

        return await history_store.get_conversation_state(str(convo_id))

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving conversation state: {e}")
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
            use_planner=body.get("use_planner", body.get("usePlanner", True)) is not False,
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
