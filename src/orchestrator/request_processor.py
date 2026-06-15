from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException, Request

from src.search import EPHEMERAL_WEB_SEARCH_EVIDENCE_MARKER

from . import proxy_services as services
from .thread_briefing import (
    build_bounded_chat_messages,
    build_thread_briefing_bundle,
)
from .trace import RequestTrace
from .utils import HeaderManager

class RequestProcessor:
    """Handles request processing, conversation management, and routing logic."""

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
    def _normalize_conversation_id(raw_conversation_id: Optional[str]) -> Optional[str]:
        conversation_id = str(raw_conversation_id or "").strip()
        return conversation_id or None

    @staticmethod
    def _normalize_reasoning_effort(raw_effort: Optional[str]) -> Optional[str]:
        effort = str(raw_effort or "").strip().lower()
        return effort if effort in services.VALID_REASONING_EFFORTS else None

    @staticmethod
    def _truthy_header(raw_value: Optional[str]) -> bool:
        return str(raw_value or "").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _normalize_retrieval_endpoint(retrieval_endpoint: Optional[str]) -> Optional[str]:
        """Normalize a configured retrieval endpoint into the external service URL."""
        if not retrieval_endpoint:
            return None

        normalized = retrieval_endpoint.strip().rstrip("/")

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
        if normalized.endswith("/api/retrieve/evidence") or normalized.endswith("/evidence"):
            raise ValueError("retrieval endpoint must target /api/retrieve/context")
        if normalized.endswith("/api/retrieve"):
            return f"{normalized}/context"
        if normalized.endswith("/context"):
            return normalized
        return f"{normalized}/api/retrieve/context"

    @staticmethod
    def _normalize_history_mode(raw_mode: Optional[str]) -> str:
        mode = str(raw_mode or "conversation").strip().lower() or "conversation"
        if mode not in {"conversation", "thread", "none"}:
            raise ValueError(f"unsupported history mode: {mode}")
        return mode

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
    def _retrieval_insertion_index(messages: List[Dict]) -> int:
        """Insert retrieval evidence immediately before the latest user turn."""
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

        return content.strip().startswith(EPHEMERAL_WEB_SEARCH_EVIDENCE_MARKER)

    @staticmethod
    def _is_ephemeral_retrieval_message(message: Dict) -> bool:
        """Identify turn-local Retrieval wrappers that must not be replay anchors."""
        if not isinstance(message, dict):
            return False

        content = message.get("content")
        if not isinstance(content, str):
            return False

        stripped = content.strip()
        return (
            stripped.startswith("Retrieved evidence excerpts:")
            or stripped.startswith("[Retrieved evidence excerpts]")
            or stripped.startswith("[Retrieved reference excerpts]")
            or stripped.startswith("[Retrieved context excerpts]")
        )

    @staticmethod
    def _is_ephemeral_conversation_message(message: Dict) -> bool:
        return RequestProcessor._is_ephemeral_search_message(
            message
        ) or RequestProcessor._is_ephemeral_retrieval_message(message)

    @staticmethod
    def _filter_ephemeral_evidence_messages(messages: List[Dict]) -> List[Dict]:
        """Drop synthetic evidence messages from durable conversation boundaries."""
        return [
            message
            for message in messages
            if not RequestProcessor._is_ephemeral_conversation_message(message)
        ]

    @staticmethod
    def public_conversation_history(messages: List[Dict]) -> List[Dict[str, str]]:
        """Return the dashboard-safe transcript: user turns and final assistant text."""
        public_messages: List[Dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            if RequestProcessor._is_ephemeral_conversation_message(message):
                continue

            role = str(message.get("role") or "").strip()
            content = message.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                continue
            if role == "assistant" and (
                message.get("tool_calls") or message.get("function_call")
            ):
                continue

            text = content.strip()
            if text:
                public_messages.append({"role": role, "content": text})
        return public_messages

    @staticmethod
    def _is_full_conversation_replay(stored: List[Dict], incoming: List[Dict]) -> bool:
        """Detect clients replaying the full server-persisted conversation prefix."""
        if not stored or len(incoming) < len(stored):
            return False
        return incoming[: len(stored)] == stored

    @staticmethod
    def _merge_ephemeral_search_evidence(messages: List[Dict]) -> List[Dict]:
        """Attach ephemeral search evidence to the next user turn for upstream chat."""
        merged: List[Dict] = []
        pending_evidence: List[str] = []

        for message in messages:
            if RequestProcessor._is_ephemeral_search_message(message):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    pending_evidence.append(content.strip())
                continue

            if pending_evidence and message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    message = dict(message)
                    message["content"] = "\n\n".join(
                        [*pending_evidence, f"Current user question:\n{content}"]
                    )
                    pending_evidence = []

            merged.append(message)

        return merged

    @staticmethod
    async def _fetch_retrieval_message(
        messages: List[Dict], retrieval_endpoint: Optional[str]
    ) -> tuple[Optional[str], Dict[str, str]]:
        """Fetch a pre-grounded user message from the retrieval service."""
        normalized_endpoint = RequestProcessor._normalize_retrieval_endpoint(retrieval_endpoint)
        if not normalized_endpoint:
            return None, {}

        latest_user_message = RequestProcessor._latest_user_message(messages)
        if not latest_user_message:
            return None, {
                "X-Retrieval-Endpoint": normalized_endpoint,
                "X-Retrieval-Injected": "false",
                "X-Retrieval-Reason": "no-user-message",
            }

        request_headers = HeaderManager.create_auth_headers()
        request_headers["Content-Type"] = "application/json"
        request_payload = {
            "query": latest_user_message,
            "limit": services.RETRIEVAL_TOP_K,
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
                "Retrieval evidence request failed for %s: %s", normalized_endpoint, exc
            )
            return None, {
                "X-Retrieval-Endpoint": normalized_endpoint,
                "X-Retrieval-Injected": "false",
                "X-Retrieval-Reason": "request-failed",
            }

        evidence_blocks = search_response.get("context_blocks")
        if not isinstance(evidence_blocks, list):
            evidence_blocks = search_response.get("evidence_blocks", [])
        if not isinstance(evidence_blocks, list):
            evidence_blocks = []

        grounded_user_message = search_response.get("grounded_user_message")
        if (
            not isinstance(grounded_user_message, str)
            or not grounded_user_message.strip()
        ):
            services.logger.info(
                "Retrieval evidence returned no grounded user message via %s",
                normalized_endpoint,
            )
            return None, {
                "X-Retrieval-Endpoint": normalized_endpoint,
                "X-Retrieval-Injected": "false",
                "X-Retrieval-Reason": "empty-grounded-user-message",
            }

        retrieval_headers = {
            "X-Retrieval-Endpoint": normalized_endpoint,
            "X-Retrieval-Hits": str(len(evidence_blocks)),
            "X-Retrieval-Injected": "true",
        }
        if search_response.get("mode") is not None:
            retrieval_headers["X-Retrieval-Mode"] = str(search_response["mode"])
        if search_response.get("truncated") is not None:
            retrieval_headers["X-Retrieval-Truncated"] = str(
                bool(search_response["truncated"])
            ).lower()

        services.logger.info(
            "Retrieval evidence retrieved %d blocks via %s (mode=%s truncated=%s)",
            len(evidence_blocks),
            normalized_endpoint,
            str(search_response.get("mode") or "unknown"),
            bool(search_response.get("truncated", False)),
        )
        return grounded_user_message.strip(), retrieval_headers

    @staticmethod
    async def prepare_request(
        request: Request,
        effective_reasoning_effort: Optional[str] = None,
        *,
        persist_conversation: bool = True,
        history_mode: Optional[str] = None,
    ) -> tuple[Dict, Dict[str, str]]:
        """Parse and enrich request with conversation history and reasoning."""
        try:
            raw_body = await request.body()
            body = json.loads(raw_body) if raw_body else {}
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON")

        if not isinstance(body, dict):
            return body, {}

        conversation_id = RequestProcessor._normalize_conversation_id(
            request.headers.get("X-Conversation-ID")
        )
        try:
            resolved_history_mode = RequestProcessor._normalize_history_mode(
                history_mode or request.headers.get("X-History-Mode")
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not persist_conversation:
            resolved_history_mode = "none"
        retrieval_endpoint = request.headers.get("X-Retrieval-Endpoint")
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
        durable_incoming_messages = []
        if conversation_id and persist_conversation:
            stored_messages = RequestProcessor._filter_ephemeral_evidence_messages(
                await services.conversation_store.get_conversation(conversation_id) or []
            )
            durable_incoming_messages = (
                RequestProcessor._filter_ephemeral_evidence_messages(incoming_messages)
            )
            if RequestProcessor._is_full_conversation_replay(
                stored_messages, durable_incoming_messages
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Full-conversation payload replay is not allowed when X-Conversation-ID is present",
                )
            if durable_incoming_messages:
                await services.conversation_store.append_messages(conversation_id, durable_incoming_messages)

        if resolved_history_mode == "conversation":
            messages = [*stored_messages, *incoming_messages]
        elif resolved_history_mode == "none" or not conversation_id:
            messages = list(incoming_messages)
        else:
            bundle = await build_thread_briefing_bundle(
                services.conversation_store,
                source_conversation_id=conversation_id,
                exclude_last_messages=len(durable_incoming_messages),
            )
            messages = build_bounded_chat_messages(
                stored_messages_before=stored_messages,
                incoming_messages=incoming_messages,
                bundle=bundle,
            )

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

        try:
            retrieval_user_content, retrieval_headers = (
                await RequestProcessor._fetch_retrieval_message(
                    messages, retrieval_endpoint
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if retrieval_user_content:
            latest_user_index = RequestProcessor._retrieval_insertion_index(messages)
            rewritten_user_message = dict(messages[latest_user_index])
            rewritten_user_message["content"] = retrieval_user_content
            messages[latest_user_index] = rewritten_user_message

        messages = RequestProcessor._merge_ephemeral_search_evidence(messages)
        body["messages"] = messages
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
            retrieval_headers["X-Reasoning-Effort"] = reasoning_effort

        return body, retrieval_headers

    @staticmethod
    async def apply_slot_affinity(
        body: Dict, conversation_id: Optional[str], endpoint_config: Optional[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Best-effort llama.cpp slot affinity for configured services.endpoints."""
        if (
            not conversation_id
            or not endpoint_config
            or not endpoint_config.get("slot_affinity")
        ):
            return {}

        endpoint = str(endpoint_config.get("name") or "").strip()
        base_url = str(endpoint_config.get("url") or "").strip().rstrip("/")
        if not endpoint or not base_url:
            return {"X-Upstream-Slot-Status": "disabled"}

        try:
            state = await services.conversation_store.get_conversation_control_state(conversation_id)
            slots = state.get("slots") if isinstance(state, dict) else {}
            slot_id = (slots or {}).get(endpoint) if isinstance(slots, dict) else None
            if slot_id is None:
                slot_id = await RequestProcessor._probe_llama_slot(base_url)
                if slot_id is None:
                    return {"X-Upstream-Slot-Status": "unavailable"}
                await services.conversation_store.set_conversation_control_slot(
                    conversation_id, endpoint, int(slot_id)
                )

            body["id_slot"] = int(slot_id)
            body["cache_prompt"] = True
            return {
                "X-Upstream-Slot-ID": str(slot_id),
                "X-Upstream-Slot-Status": "affinity-applied",
            }
        except Exception as exc:
            services.logger.info("Slot affinity skipped for %s/%s: %s", endpoint, conversation_id, exc)
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
    async def update_conversation(
        conversation_id: Optional[str], assistant_text: Optional[str]
    ) -> Dict[str, Any]:
        """Persist assistant response text and report the durable outcome."""
        text = assistant_text or ""
        outcome: Dict[str, Any] = {
            "assistant_chars": len(text),
            "assistant_persisted": False,
        }
        if not conversation_id:
            outcome["assistant_skip_reason"] = "no-conversation-id"
            return outcome
        if not text:
            outcome["assistant_skip_reason"] = "empty-assistant"
            return outcome

        await services.conversation_store.append_messages(
            conversation_id, [{"role": "assistant", "content": text}]
        )
        outcome["assistant_persisted"] = True
        if RequestProcessor._schedule_thread_state_refresh(conversation_id):
            outcome["thread_state_refresh_scheduled"] = True
        return outcome

    @staticmethod
    def _schedule_thread_state_refresh(conversation_id: str) -> bool:
        if not RequestProcessor._truthy_header(
            os.getenv("THREAD_STATE_REFRESH_ENABLED")
        ):
            return False
        endpoint = str(os.getenv("THREAD_STATE_REFRESH_ENDPOINT") or "").strip()
        if not endpoint:
            services.logger.warning(
                "Thread state refresh enabled but THREAD_STATE_REFRESH_ENDPOINT is unset"
            )
            return False

        from .thread_state_refresh import maybe_refresh_thread_state
        from .workflow_clients import ProxyWorkflowLLMClient

        task = asyncio.create_task(
            maybe_refresh_thread_state(
                conversation_store=services.conversation_store,
                llm_client=ProxyWorkflowLLMClient(),
                source_conversation_id=conversation_id,
                endpoint=endpoint,
                reasoning_effort=RequestProcessor._normalize_reasoning_effort(
                    os.getenv("THREAD_STATE_REFRESH_REASONING_EFFORT")
                ),
                force=False,
            )
        )
        services.track_conversation_finalization(task)
        return True

    @staticmethod
    async def finalize_stream_conversation(
        conversation_id: Optional[str],
        assistant_text: Optional[str],
        trace: Optional[RequestTrace],
    ) -> None:
        """Persist streamed assistant text and emit completion trace metadata."""
        try:
            conversation_outcome = await RequestProcessor.update_conversation(
                conversation_id, assistant_text
            )
            if trace:
                trace.conversation.update(conversation_outcome)
        except Exception as exc:
            services.logger.exception(
                "Failed to persist assistant response for conversation_id=%s",
                conversation_id,
            )
            if trace:
                trace.error_class = trace.error_class or type(exc).__name__
                trace.conversation.update(
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
        conversation_id: Optional[str],
        assistant_text: Optional[str],
        trace: Optional[RequestTrace],
    ) -> None:
        """Run stream finalization in a task that survives caller cancellation."""
        task = services.track_conversation_finalization(
            asyncio.create_task(
                RequestProcessor.finalize_stream_conversation(
                    conversation_id, assistant_text, trace
                )
            )
        )
        await asyncio.shield(task)
