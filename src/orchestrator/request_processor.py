from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException, Request

from src.search import EPHEMERAL_WEB_SEARCH_CONTEXT_MARKER

from . import proxy_services as services
from .trace import RequestTrace
from .utils import HeaderManager

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
            for endpoint in services.endpoints
            if str(endpoint.get("name") or "").strip()
        ]

    @staticmethod
    def get_endpoint_config(path_or_endpoint: str) -> Optional[Dict[str, Any]]:
        """Return endpoint config by path or endpoint name."""
        endpoint_key = RequestProcessor._endpoint_key(path_or_endpoint)
        for endpoint_config in services.endpoints:
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

        services.logger.warning(f"No endpoint found for path: {path}")
        return None

    @staticmethod
    def _normalize_convo_id(raw_convo_id: Optional[str]) -> Optional[str]:
        convo_id = str(raw_convo_id or "").strip()
        return convo_id or None

    @staticmethod
    def _normalize_reasoning_effort(raw_effort: Optional[str]) -> Optional[str]:
        effort = str(raw_effort or "").strip().lower()
        return effort if effort in services.VALID_REASONING_EFFORTS else None

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
            "limit": services.RAG_TOP_K,
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    normalized_endpoint, json=request_payload, headers=request_headers
                )
                response.raise_for_status()
                search_response = response.json()
        except Exception as exc:
            services.logger.warning(
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
            services.logger.info(
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

        services.logger.info(
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
                await services.history_store.get_conversation(convo_id) or []
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
                await services.history_store.append_messages(convo_id, durable_incoming_messages)

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
                services.logger.info(f"Applied reasoning: {reasoning_effort}")

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
        """Best-effort llama.cpp slot affinity for configured services.endpoints."""
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
            state = await services.history_store.get_conversation_state(convo_id)
            slots = state.get("slots") if isinstance(state, dict) else {}
            slot_id = (slots or {}).get(endpoint) if isinstance(slots, dict) else None
            if slot_id is None:
                slot_id = await RequestProcessor._probe_llama_slot(base_url)
                if slot_id is None:
                    return {"X-Upstream-Slot-Status": "unavailable"}
                await services.history_store.set_conversation_slot(
                    convo_id, endpoint, int(slot_id)
                )

            body["id_slot"] = int(slot_id)
            body["cache_prompt"] = True
            return {
                "X-Upstream-Slot-ID": str(slot_id),
                "X-Upstream-Slot-Status": "affinity-applied",
            }
        except Exception as exc:
            services.logger.info("Slot affinity skipped for %s/%s: %s", endpoint, convo_id, exc)
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

        await services.history_store.append_messages(
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
            services.logger.exception(
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
                services.logger.exception("Failed to emit request trace")

    @staticmethod
    async def finalize_stream_response(
        convo_id: Optional[str],
        assistant_text: Optional[str],
        trace: Optional[RequestTrace],
    ) -> None:
        """Run stream finalization in a task that survives caller cancellation."""
        task = services.track_history_finalization(
            asyncio.create_task(
                RequestProcessor.finalize_stream_history(
                    convo_id, assistant_text, trace
                )
            )
        )
        await asyncio.shield(task)
