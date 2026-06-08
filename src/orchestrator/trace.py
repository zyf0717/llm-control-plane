"""Request trace metadata skeleton."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


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
    timing: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    _started_at: float = field(default_factory=time.perf_counter, repr=False)

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

    def mark_elapsed(self) -> None:
        """Record elapsed request time in milliseconds."""
        self.timing["elapsed_ms"] = max(
            0,
            int((time.perf_counter() - self._started_at) * 1000),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable trace payload."""
        return {
            "request_id": self.request_id,
            "convo_id": self.convo_id,
            "endpoint": self.endpoint,
            "route": dict(self.route),
            "search": dict(self.search),
            "rag": dict(self.rag),
            "timing": dict(self.timing),
            "warnings": list(self.warnings),
        }
