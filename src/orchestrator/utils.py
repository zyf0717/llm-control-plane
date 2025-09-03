import json
import logging
import os
from typing import Dict, Optional

from fastapi import Request

logger = logging.getLogger(__name__)

START_REASONING_SEEDS = ("<|channel|>", "<think>")
START_REASONING_SEQ = ("<|channel|>analysis<|message|>", "<think>")
END_REASONING_SEEDS = ("<|end|>", "</think>")
END_REASONING_SEQ = ("</think>", "<|end|><|start|>assistant<|channel|>final<|message|>")


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
        headers = {}
        api_key_id = os.getenv("API_KEY_ID")
        api_key_secret = os.getenv("API_KEY_SECRET")

        if api_key_id:
            headers["CF-Access-Client-Id"] = api_key_id
        if api_key_secret:
            headers["CF-Access-Client-Secret"] = api_key_secret

        return headers

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


async def process_stream_line(
    line: str,
    acc: SSEAccumulator,
    start_reasoning_buffer: str,
    end_reasoning_buffer: str,
) -> tuple[Optional[bytes], str, str, bool]:
    """
    Process a single streaming response line with reasoning channel detection.

    Returns:
        tuple: (chunk_to_yield, new_start_buffer, new_end_buffer, should_continue)
    """
    # 1. Format is "data: {json}""
    if not line or not line.startswith("data: "):
        return None, start_reasoning_buffer, end_reasoning_buffer, True

    data = line[6:].strip()

    # 2. End loop if [DONE] is received
    if data == "[DONE]":
        chunk = (f"data: {data}\n\n").encode("utf-8")
        return chunk, start_reasoning_buffer, end_reasoning_buffer, False

    # 3. Parse delta and content, fall back to passthrough if JSON malformed
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        chunk = (line + "\n\n").encode("utf-8")
        acc.feed(chunk)
        return chunk, start_reasoning_buffer, end_reasoning_buffer, True
    delta = (obj.get("choices") or [{}])[0].get("delta") or {}
    content = delta.get("content", "")

    # 4. If no content (e.g., tool/role deltas), passthrough unchanged
    if not content:
        chunk = line.encode("utf-8") + b"\n\n"
        acc.feed(chunk)
        return chunk, start_reasoning_buffer, end_reasoning_buffer, True

    # 5. Channel start/end detection to start respective buffers
    if content in START_REASONING_SEEDS and not start_reasoning_buffer:
        logger.debug("start_reasoning_buffer: %s", content)
        return None, content, end_reasoning_buffer, True
    elif content in END_REASONING_SEEDS and not end_reasoning_buffer:
        logger.debug("end_reasoning_buffer: %s", content)
        return None, start_reasoning_buffer, content, True

    # 6. If end of reasoning/analysis channel found, clear buffers, proceed to yield in step 10.
    if end_reasoning_buffer in END_REASONING_SEQ:
        logger.debug("Switching to content channel...")
        start_reasoning_buffer = ""
        end_reasoning_buffer = ""
        # Note: no `continue` here in order to yield first content message

    # 7. Otherwise if end of reasoning/analysis buffer not empty, accumulate
    elif end_reasoning_buffer:
        end_reasoning_buffer += content
        logger.debug(
            "Accumulating end_reasoning_buffer: %s",
            end_reasoning_buffer,
        )
        return None, start_reasoning_buffer, end_reasoning_buffer, True

    # 8. If reasoning/analysis channel found, yield into a "reasoning" channel
    if start_reasoning_buffer in START_REASONING_SEQ:
        logger.debug("Reasoning stream: %s", content)
        obj["choices"][0]["delta"].pop("content", None)
        obj["choices"][0]["delta"]["reasoning"] = content
        chunk = f"data: {json.dumps(obj)}".encode("utf-8") + b"\n\n"
        return chunk, start_reasoning_buffer, end_reasoning_buffer, True

    # 9. If start of reasoning/analysis buffer not empty, accumulate
    elif start_reasoning_buffer:
        start_reasoning_buffer += content
        logger.debug(
            "Accumulating start_reasoning_buffer: %s",
            start_reasoning_buffer,
        )
        return None, start_reasoning_buffer, end_reasoning_buffer, True

    # 10. Yield content as-is if both buffers are empty (i.e. default-path)
    if not start_reasoning_buffer and not end_reasoning_buffer:
        logger.debug("Content stream: %s", content)
        chunk = line.encode("utf-8") + b"\n\n"
        acc.feed(chunk)
        return chunk, start_reasoning_buffer, end_reasoning_buffer, True

    return None, start_reasoning_buffer, end_reasoning_buffer, True


def process_non_stream_response(resp_json: Dict) -> Dict:
    """
    Process non-streaming response to extract reasoning content to separate channel.

    Returns:
        Dict: Modified response with reasoning extracted to separate field
    """
    try:
        # Get the assistant message content
        choices = resp_json.get("choices", [])
        if not choices:
            return resp_json

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "")

        if not content:
            return resp_json

        reasoning = ""
        clean_content = content

        # Process <think> tags - extract and remove
        if "<think>" in content and "</think>" in content:
            # Simple split and join approach
            parts = content.split("<think>")
            clean_parts = [parts[0]]  # Content before first <think> (could be empty)
            reasoning_parts = []

            for part in parts[1:]:
                if "</think>" in part:
                    think_part, after_part = part.split("</think>", 1)
                    reasoning_parts.append(think_part)
                    clean_parts.append(after_part)
                else:
                    # No closing tag, keep as is
                    clean_parts.append("<think>" + part)

            if reasoning_parts:
                reasoning = "\n\n".join(reasoning_parts).strip()
                clean_content = "".join(clean_parts).strip()

        # Process <|channel|> tags - extract and remove
        elif "<|channel|>analysis<|message|>" in content:
            # Split by analysis start
            before, analysis_part = content.split("<|channel|>analysis<|message|>", 1)

            # Split by analysis end
            if "<|end|><|start|>assistant<|channel|>final<|message|>" in analysis_part:
                reasoning_content, after = analysis_part.split(
                    "<|end|><|start|>assistant<|channel|>final<|message|>", 1
                )
                reasoning = reasoning_content.strip()
                # Handle edge cases: before and/or after could be empty
                clean_content = (before + after).strip() or ""

        # Return modified response if reasoning was found
        if reasoning:
            modified_resp = resp_json.copy()
            modified_resp["choices"] = [choice.copy() for choice in choices]
            modified_resp["choices"][0]["message"] = message.copy()
            # Ensure clean_content is never None, use empty string if no content remains
            modified_resp["choices"][0]["message"]["content"] = clean_content or ""
            modified_resp["choices"][0]["message"]["reasoning"] = reasoning
            return modified_resp

        return resp_json

    except Exception as e:
        logger.error("Error processing non-stream response: %s", str(e))
        return resp_json
