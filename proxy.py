import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from utils import (
    SSEAccumulator,
    create_error_sse_message,
    extract_assistant_text,
    filter_unsafe_headers,
)

load_dotenv()
API_KEY_ID = os.getenv("API_KEY_ID")
API_KEY_SECRET = os.getenv("API_KEY_SECRET")
GPT_OSS_20B_API_URL = os.getenv("GPT_OSS_20B_API_URL")
QWEN_3_4B_API_URL = os.getenv("QWEN_3_4B_API_URL")

# Endpoint mapping
ENDPOINT_MAP = {
    "gpt-oss-20b": f"{GPT_OSS_20B_API_URL}/v1/chat/completions",
    "gpt-oss-20b-api": f"{GPT_OSS_20B_API_URL}/api/v0/chat/completions",
    "qwen3-4b": f"{QWEN_3_4B_API_URL}/v1/chat/completions",
    "qwen3-4b-api": f"{QWEN_3_4B_API_URL}/api/v0/chat/completions",
}
DEFAULT_ENDPOINT = f"{GPT_OSS_20B_API_URL}/v1/chat/completions"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
app = FastAPI()

# Simple global store (not persistent across restarts)
convo_history: Dict[str, List[Dict]] = {}


def prepare_headers(request: Request) -> Dict[str, str]:
    """Prepare headers for upstream request."""
    headers = filter_unsafe_headers(dict(request.headers))
    headers["CF-Access-Client-Id"] = API_KEY_ID
    headers["CF-Access-Client-Secret"] = API_KEY_SECRET
    return headers


def get_target_endpoint(path: str) -> str:
    """Get target endpoint URL based on path."""
    return ENDPOINT_MAP.get(path, DEFAULT_ENDPOINT)


def parse_and_inject_history(
    body: bytes, convo_id: Optional[str]
) -> Tuple[bytes, bool]:
    """Parse request body and inject conversation history if applicable."""
    if not body:
        return body, False

    try:
        body_json = json.loads(body)
        if not isinstance(body_json, dict):
            return body, False

        is_streaming = body_json.get("stream", False)

        # Inject history if messages exist and convo_id provided
        if (
            "messages" in body_json
            and isinstance(body_json["messages"], list)
            and convo_id
        ):
            if convo_id not in convo_history:
                convo_history[convo_id] = []

            # Append new messages to history
            convo_history[convo_id].extend(body_json["messages"])

            # Replace payload messages with full history
            body_json["messages"] = convo_history[convo_id]

        return json.dumps(body_json).encode(), is_streaming

    except Exception:
        logger.warning("Failed to parse request body as JSON")
        return body, False


def update_conversation_history(convo_id: Optional[str], assistant_text: str) -> None:
    """Update conversation history with assistant response."""
    if assistant_text and convo_id and convo_history.get(convo_id) is not None:
        convo_history[convo_id].append({"role": "assistant", "content": assistant_text})


def log_response_info(resp_json: Dict) -> Optional[str]:
    """Log response information and extract assistant text."""
    # Standard logs
    if "created" in resp_json:
        ts = datetime.fromtimestamp(resp_json["created"])
        logger.info("Response created: %s", ts)
    if "model" in resp_json:
        logger.info("Response model: %s", resp_json["model"])
    if "usage" in resp_json:
        logger.info("Usage: %s", resp_json["usage"])

    assistant_text = extract_assistant_text(resp_json)

    finish_reason = (resp_json.get("choices") or [{}])[0].get("finish_reason")
    if finish_reason:
        logger.info("Finish reason: %s", finish_reason)

    return assistant_text


def create_safe_headers(resp_headers: Dict, convo_id: Optional[str]) -> Dict[str, str]:
    """Create safe response headers."""
    safe_headers = filter_unsafe_headers(resp_headers)
    if convo_id:
        safe_headers["X-Convo-ID"] = convo_id
    return safe_headers


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/")
async def root_chat(request: Request):
    """Route root POST requests directly to chat/completions."""
    return await proxy_with_context("", request)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def custom_endpoints(path: str, request: Request):
    """Handle all other endpoints."""
    if path == "health":
        return await health_check()
    return await proxy_with_context(path, request)


async def handle_streaming_response(
    request: Request,
    target_endpoint: str,
    headers: Dict[str, str],
    body: bytes,
    convo_id: Optional[str],
) -> StreamingResponse:
    """Handle streaming response proxying."""
    timeout = httpx.Timeout(connect=20, read=None, write=20, pool=20)
    upstream_headers = {
        k: v for k, v in headers.items() if k.lower() not in {"content-length", "host"}
    }
    upstream_headers["Accept"] = "text/event-stream"

    acc = SSEAccumulator()

    async def stream_response():
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    method=request.method,
                    url=target_endpoint,
                    headers=upstream_headers,
                    content=body if body else None,
                    params=dict(request.query_params),
                ) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_bytes():
                        acc.feed(chunk)
                        yield chunk
        except httpx.HTTPStatusError as e:
            yield create_error_sse_message(
                "error", status=e.response.status_code, detail=e.response.text
            )
        except Exception as e:
            yield create_error_sse_message("error", detail=repr(e))
        finally:
            # Update conversation history with assembled response
            try:
                assembled = acc.text()
                update_conversation_history(convo_id, assembled)
            except Exception:
                pass  # Don't let history failures affect client stream

    resp_headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    if convo_id:
        resp_headers["X-Convo-ID"] = convo_id

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers=resp_headers,
    )


async def handle_non_streaming_response(
    request: Request,
    target_endpoint: str,
    headers: Dict[str, str],
    body: bytes,
    convo_id: Optional[str],
) -> Response:
    """Handle non-streaming response proxying."""
    async with httpx.AsyncClient() as client:
        resp = await client.request(
            method=request.method,
            url=target_endpoint,
            headers=headers,
            content=body if body else None,
            params=dict(request.query_params),
            timeout=120,
        )

        assistant_text = None
        try:
            resp_json = resp.json()
            assistant_text = log_response_info(resp_json)
        except Exception:
            logger.info("Response: %s", resp.text[-500:])

        # Update conversation history
        update_conversation_history(convo_id, assistant_text)

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=create_safe_headers(dict(resp.headers), convo_id),
        )


async def proxy_with_context(path: str, request: Request):
    """Main proxy handler with conversation context."""
    headers = prepare_headers(request)
    target_endpoint = get_target_endpoint(path)

    # Parse request body and inject conversation history
    body = await request.body()
    convo_id = request.headers.get("X-Convo-ID")
    if convo_id:
        headers["X-Convo-ID"] = convo_id

    body, is_streaming = parse_and_inject_history(body, convo_id)

    # Check for streaming flag in query params as fallback
    if not is_streaming:
        is_streaming = request.query_params.get("stream") in {"true", "1"}

    logger.info(
        "Proxying %s %s to %s (streaming: %s)",
        request.method,
        request.url.path,
        target_endpoint,
        is_streaming,
    )

    if is_streaming:
        return await handle_streaming_response(
            request, target_endpoint, headers, body, convo_id
        )
    else:
        return await handle_non_streaming_response(
            request, target_endpoint, headers, body, convo_id
        )
