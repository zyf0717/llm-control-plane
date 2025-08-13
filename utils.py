import json


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
