import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from utils import (
    HeaderManager,
    SSEAccumulator,
    create_error_sse_message,
    extract_assistant_text,
)

# Load environment variables
load_dotenv()

CONFIG_FILE = Path("config.yaml")
with CONFIG_FILE.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}

endpoints = config.get("endpoints", [])


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
app = FastAPI()

# Simple global store (not persistent across restarts)
convo_history: Dict[str, List[Dict]] = {}


def get_target_endpoint(path: str, is_streaming: bool = False) -> str:
    """Get the target endpoint URL based on the request path and streaming preference."""
    # Remove leading slash and get the first path segment
    clean_path = path.lstrip("/")
    endpoint_key = clean_path.split("/")[0] if clean_path else ""

    # Find the matching endpoint in config.yaml
    for endpoint_config in endpoints:
        endpoint_name = endpoint_config.get("name")
        if endpoint_name == endpoint_key:
            endpoint_url = endpoint_config.get("url")
            if endpoint_url:
                # Route based on streaming preference
                if is_streaming:
                    # OpenAI-style streaming endpoint with telemetrics
                    return f"{endpoint_url}/v1/chat/completions"
                else:
                    # Non-streaming API endpoint
                    return f"{endpoint_url}/api/v0/chat/completions"

    # If no endpoint found, return None and let the caller handle the error
    logger.warning(f"No endpoint found for path: {path}")
    return None


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


@app.post("/")
async def root_chat(request: Request):
    """Route root POST requests directly to chat/completions."""
    return await proxy_with_context("", request)


@app.get("/models")
async def list_models():
    """List all available models from configured endpoints with additional metadata."""
    available_models = []

    # Iterate through all configured endpoints
    for endpoint_config in endpoints:
        endpoint_name = endpoint_config.get("name", "unknown")
        endpoint_url = endpoint_config.get("url")

        if not endpoint_url:
            logger.warning(f"No URL configured for endpoint {endpoint_name}")
            continue

        try:
            # Send GET request to /api/v0/models endpoint
            models_url = f"{endpoint_url}/api/v0/models"
            headers = HeaderManager.create_auth_headers()

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(models_url, headers=headers)
                response.raise_for_status()

                models_data = response.json()

                # Extract models from response (handle both direct list and nested object)
                if isinstance(models_data, dict) and "data" in models_data:
                    remote_models = models_data["data"]
                elif isinstance(models_data, dict) and "models" in models_data:
                    remote_models = models_data["models"]
                elif isinstance(models_data, list):
                    remote_models = models_data
                else:
                    logger.warning(
                        f"Unexpected models response format from {endpoint_name}"
                    )
                    continue

                # Process each model and inject additional metadata
                for model in remote_models:
                    if isinstance(model, dict):
                        # Create enhanced model object with OpenAI-compatible fields
                        enhanced_model = {
                            "id": model.get("id", f"unknown-{endpoint_name}"),
                            "object": "model",
                            "created": model.get(
                                "created", int(datetime.now().timestamp())
                            ),
                            "owned_by": model.get("owned_by", endpoint_name),
                            # Inject additional metadata from config
                            "endpoint": endpoint_name,
                            "endpoint_url": endpoint_url,
                        }

                        # Add hardware specs if available in config
                        if "gpu" in endpoint_config:
                            enhanced_model["gpu"] = endpoint_config["gpu"]
                        if "vram" in endpoint_config:
                            enhanced_model["vram"] = endpoint_config["vram"]
                        if "soc" in endpoint_config:
                            enhanced_model["soc"] = endpoint_config["soc"]
                        if "cpu" in endpoint_config:
                            enhanced_model["cpu"] = endpoint_config["cpu"]
                        if "ram" in endpoint_config:
                            enhanced_model["ram"] = endpoint_config["ram"]

                        # Preserve any additional fields from the original model response
                        for key, value in model.items():
                            if key not in enhanced_model:
                                enhanced_model[key] = value

                        available_models.append(enhanced_model)

        except httpx.HTTPError as e:
            logger.warning(
                f"Failed to fetch models from {endpoint_name} ({endpoint_url}): {e}"
            )
        except Exception as e:
            logger.error(f"Unexpected error fetching models from {endpoint_name}: {e}")

    # Return in OpenAI-compatible format
    return {"object": "list", "data": available_models}


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def custom_endpoints(path: str, request: Request):
    """Handle all other endpoints."""
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

    # Prepare streaming options
    body["stream_options"] = {"include_usage": True}

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
    # Parse request body and inject conversation history first
    raw_body = await request.body()
    convo_id = request.headers.get("X-Convo-ID")

    body, is_streaming = parse_and_inject_history(raw_body, convo_id)

    # Check for streaming flag in query params as fallback
    if not is_streaming:
        is_streaming = request.query_params.get("stream") in {"true", "1"}

    # Get target endpoint based on streaming preference
    target_endpoint = get_target_endpoint(path, is_streaming)

    if not target_endpoint:
        raise HTTPException(
            status_code=404, detail=f"Endpoint not found for path: {path}"
        )

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
