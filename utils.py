import json
import logging
from typing import Dict, Optional

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
