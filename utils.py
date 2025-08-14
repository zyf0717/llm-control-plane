import json
import logging
import os
from typing import Dict, Optional

from fastapi import Request

logger = logging.getLogger(__name__)


class SSEAccumulator:
    def __init__(self):
        self._parts = []

    def feed(self, chunk: bytes):
        # Parse SSE "data: {...}" lines and accumulate content deltas
        for line in chunk.split(b"\n"):
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if not payload or payload == b"[DONE]":
                continue
            try:
                obj = json.loads(payload)
                ch0 = (obj.get("choices") or [None])[0] or {}
                delta = ch0.get("delta") or {}
                frag = delta.get("content")
                if frag:
                    self._parts.append(frag)
            except Exception:
                # swallow parsing errors; passthrough must not break
                pass

    def text(self) -> str | None:
        return "".join(self._parts) if self._parts else None


def extract_assistant_text(resp_json: Dict) -> Optional[str]:
    """Extract assistant text from response JSON (OpenAI-style)."""
    ch0 = (resp_json.get("choices") or [None])[0] or {}
    # New-style
    if isinstance(ch0.get("message"), dict):
        return ch0["message"].get("content")
    # v0/legacy fallback
    return ch0.get("text")


def create_error_sse_message(error_type: str, **kwargs) -> bytes:
    """Create SSE error message."""
    data = {"type": f"proxy.{error_type}", **kwargs}
    msg = f"data: {json.dumps(data)}\n\n"
    return msg.encode("utf-8")


def filter_unsafe_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Filter out headers that shouldn't be passed through."""
    unsafe_headers = {
        "content-encoding",
        "transfer-encoding",
        "connection",
        "host",
        "content-length",
    }
    return {k: v for k, v in headers.items() if k.lower() not in unsafe_headers}


class HeaderManager:
    """Centralized header management for the proxy."""

    @staticmethod
    def create_auth_headers() -> Dict[str, str]:
        """Create authentication headers for upstream requests."""
        return {
            "CF-Access-Client-Id": os.getenv("API_KEY_ID"),
            "CF-Access-Client-Secret": os.getenv("API_KEY_SECRET"),
        }

    @staticmethod
    def prepare_upstream_headers(
        request: Request, for_streaming: bool = False
    ) -> Dict[str, str]:
        """Prepare headers for upstream requests."""
        headers = filter_unsafe_headers(dict(request.headers))
        headers.update(HeaderManager.create_auth_headers())

        if for_streaming:
            # Remove content-length for streaming and set proper accept header
            headers.pop("content-length", None)
            headers["Accept"] = "text/event-stream"

        return headers

    @staticmethod
    def create_response_headers(
        source_headers: Optional[Dict] = None,
        convo_id: Optional[str] = None,
        for_streaming: bool = False,
    ) -> Dict[str, str]:
        """Create safe response headers."""
        if source_headers:
            headers = filter_unsafe_headers(source_headers)
        else:
            headers = {}

        if for_streaming:
            headers.update(
                {
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
            )

        if convo_id:
            headers["X-Convo-ID"] = convo_id

        return headers
