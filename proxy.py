import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from utils import (
    HeaderManager,
    SSEAccumulator,
    create_error_sse_message,
    extract_assistant_text,
)

load_dotenv()
GPT_OSS_20B_API_URL = os.getenv("GPT_OSS_20B_API_URL")
QWEN_3_4B_API_URL = os.getenv("QWEN_3_4B_API_URL")

# Endpoint mapping
ENDPOINT_MAP = {
    "gpt-oss-20b-api": f"{GPT_OSS_20B_API_URL}/api/v0/chat/completions",
    "qwen3-4b-api": f"{QWEN_3_4B_API_URL}/api/v0/chat/completions",
    "gpt-oss-20b": f"{GPT_OSS_20B_API_URL}/v1/chat/completions",
    "qwen3-4b": f"{QWEN_3_4B_API_URL}/v1/chat/completions",
}
DEFAULT_ENDPOINT = f"{GPT_OSS_20B_API_URL}/v1/chat/completions"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
app = FastAPI()

# Simple global store (not persistent across restarts)
convo_history: Dict[str, List[Dict]] = {}

# Cache for endpoint health checks (endpoint_url -> (is_online, timestamp))
endpoint_health_cache: Dict[str, Tuple[bool, float]] = {}
HEALTH_CACHE_TTL = 30  # seconds


async def check_endpoint_health(endpoint_url: str, timeout: float = 5.0) -> bool:
    """Check if an endpoint is healthy by sending a GET request with proper auth headers."""
    current_time = time.time()

    # Check cache first
    if endpoint_url in endpoint_health_cache:
        is_healthy, timestamp = endpoint_health_cache[endpoint_url]
        if current_time - timestamp < HEALTH_CACHE_TTL:
            return is_healthy

    # Cache miss or expired, perform actual check
    try:
        headers = HeaderManager.create_auth_headers()
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            response = await client.get(endpoint_url, headers=headers)
            # Consider 2xx and 4xx as healthy (4xx means auth/client issues, not server down)
            # 5xx means server errors (unhealthy)
            is_healthy = 200 <= response.status_code < 500
    except Exception:
        is_healthy = False

    # Update cache
    endpoint_health_cache[endpoint_url] = (is_healthy, current_time)
    return is_healthy


def get_target_endpoint(path: str) -> str:
    """Get the target endpoint URL based on the request path."""
    # Remove leading slash and get the first path segment
    clean_path = path.lstrip("/")
    endpoint_key = clean_path.split("/")[0] if clean_path else ""

    # Return mapped endpoint or default
    return ENDPOINT_MAP.get(endpoint_key, DEFAULT_ENDPOINT)


async def get_available_endpoint(path: str) -> str:
    """Get an available endpoint, falling back to alternatives if primary is offline."""
    primary_endpoint = get_target_endpoint(path)

    # First try the primary endpoint
    if await check_endpoint_health(primary_endpoint):
        return primary_endpoint

    logger.warning(
        "Primary endpoint %s is offline, trying alternatives", primary_endpoint
    )

    # Try other endpoints as fallbacks
    for endpoint_url in ENDPOINT_MAP.values():
        if endpoint_url != primary_endpoint and await check_endpoint_health(
            endpoint_url
        ):
            logger.info("Using fallback endpoint: %s", endpoint_url)
            return endpoint_url

    # If all else fails, return the default (let it fail downstream)
    logger.error("All endpoints appear to be offline, using default")
    return DEFAULT_ENDPOINT


def parse_and_inject_history(
    body: bytes, convo_id: Optional[str]
) -> Tuple[Optional[Dict], bool]:
    """Parse request body and inject conversation history if applicable."""
    if not body:
        return None, False

    try:
        body_json = json.loads(body)
        if not isinstance(body_json, dict):
            return None, False

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

        return body_json, is_streaming

    except Exception:
        logger.warning("Failed to parse request body as JSON")
        return None, False


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


@app.get("/health")
async def health_check():
    """Health check endpoint that also reports endpoint statuses."""
    endpoint_statuses = {}

    # Check status of all configured endpoints
    for name, url in ENDPOINT_MAP.items():
        endpoint_statuses[name] = await check_endpoint_health(url, timeout=2.0)

    # Overall health is OK if at least one endpoint is available
    overall_healthy = any(endpoint_statuses.values())

    return {
        "status": "ok" if overall_healthy else "degraded",
        "endpoints": endpoint_statuses,
    }


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
    body: Optional[Dict],
    convo_id: Optional[str],
) -> StreamingResponse:
    """Handle streaming response proxying."""
    timeout = httpx.Timeout(connect=20, read=None, write=20, pool=20)
    upstream_headers = HeaderManager.prepare_upstream_headers(
        request, for_streaming=True
    )

    acc = SSEAccumulator()

    async def stream_response():
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    method=request.method,
                    url=target_endpoint,
                    headers=upstream_headers,
                    json=body,
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

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers=HeaderManager.create_response_headers(
            convo_id=convo_id, for_streaming=True
        ),
    )


async def handle_non_streaming_response(
    request: Request,
    target_endpoint: str,
    body: Optional[Dict],
    convo_id: Optional[str],
) -> Response:
    """Handle non-streaming response proxying."""
    upstream_headers = HeaderManager.prepare_upstream_headers(request)

    async with httpx.AsyncClient() as client:
        resp = await client.request(
            method=request.method,
            url=target_endpoint,
            headers=upstream_headers,
            json=body,
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
            headers=HeaderManager.create_response_headers(dict(resp.headers), convo_id),
        )


async def proxy_with_context(path: str, request: Request):
    """Main proxy handler with conversation context."""
    target_endpoint = await get_available_endpoint(path)

    # Parse request body and inject conversation history
    raw_body = await request.body()
    convo_id = request.headers.get("X-Convo-ID")

    body, is_streaming = parse_and_inject_history(raw_body, convo_id)

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
        return await handle_streaming_response(request, target_endpoint, body, convo_id)
    else:
        return await handle_non_streaming_response(
            request, target_endpoint, body, convo_id
        )
