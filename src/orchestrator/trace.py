"""Request trace metadata and structured JSONL persistence."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from src.logging_config import get_trace_log_path
from src.search.safety import EPHEMERAL_WEB_SEARCH_CONTEXT_MARKER


TRACE_ID_HEADER = "X-Trace-ID"


@dataclass(slots=True)
class RequestTrace:
    """Minimal per-request trace object for future observability."""

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    convo_id: Optional[str] = None
    endpoint: Optional[str] = None
    route: dict[str, str] = field(default_factory=dict)
    search: dict[str, Any] = field(default_factory=dict)
    rag: dict[str, str] = field(default_factory=dict)
    slot: dict[str, str] = field(default_factory=dict)
    history: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    status_code: Optional[int] = None
    error_class: Optional[str] = None
    _started_at: float = field(default_factory=time.perf_counter, repr=False)
    _emitted_phases: set[str] = field(default_factory=set, init=False, repr=False)

    def apply_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Attach trace headers to a response header mapping."""
        headers[TRACE_ID_HEADER] = self.request_id
        return headers

    def capture_headers(self, headers: dict[str, Any]) -> None:
        """Copy route/RAG response headers into normalized trace fields."""
        for name, value in headers.items():
            lower_name = str(name).lower()
            if lower_name.startswith("x-route-"):
                self.route[lower_name.replace("x-route-", "")] = str(value)
            elif lower_name.startswith("x-rag-"):
                self.rag[lower_name.replace("x-rag-", "")] = str(value)
            elif lower_name.startswith("x-upstream-slot-"):
                self.slot[lower_name.replace("x-upstream-slot-", "")] = str(value)

    def capture_search_from_body(self, body: dict[str, Any]) -> None:
        """Record presence of turn-local web search context without storing content."""
        messages = body.get("messages") if isinstance(body, dict) else None
        if not isinstance(messages, list):
            return

        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            if EPHEMERAL_WEB_SEARCH_CONTEXT_MARKER in content:
                self.search["injected"] = "true"
                self._capture_search_metadata(content)
                return

    def _capture_search_metadata(self, content: str) -> None:
        payload_text = content.split(EPHEMERAL_WEB_SEARCH_CONTEXT_MARKER, 1)[1].strip()
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return

        provider = str(payload.get("provider") or "").strip()
        if provider:
            self.search["provider"] = provider
        results = payload.get("results")
        if isinstance(results, list):
            self.search["result_count"] = len(results)
        if "degraded" in payload:
            self.search["degraded"] = bool(payload.get("degraded"))
        warnings = payload.get("warnings")
        if isinstance(warnings, list):
            self.search["warning_count"] = len(warnings)

    def mark_elapsed(self) -> None:
        """Record elapsed request time in milliseconds."""
        self.timing["elapsed_ms"] = max(
            0,
            int((time.perf_counter() - self._started_at) * 1000),
        )

    def to_dict(self, *, phase: str = "completed") -> dict[str, Any]:
        """Return a JSON-serializable trace payload."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "request_id": self.request_id,
            "convo_id": self.convo_id,
            "endpoint": self.endpoint,
            "route": dict(self.route),
            "search": dict(self.search),
            "rag": dict(self.rag),
            "slot": dict(self.slot),
            "history": dict(self.history),
            "timing": dict(self.timing),
            "warnings": list(self.warnings),
            "status_code": self.status_code,
            "error_class": self.error_class,
        }

    def emit(
        self,
        *,
        status_code: Optional[int] = None,
        error: Any = None,
        phase: str = "completed",
    ) -> None:
        """Append this trace as one sanitized JSONL event."""
        normalized_phase = str(phase or "completed").strip() or "completed"
        if normalized_phase in self._emitted_phases:
            return

        if status_code is not None:
            self.status_code = int(status_code)
        if error is not None:
            self.error_class = type(error).__name__
        self.mark_elapsed()

        path = get_trace_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            self.to_dict(phase=normalized_phase),
            sort_keys=True,
            ensure_ascii=False,
        )
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")
        self._emitted_phases.add(normalized_phase)
