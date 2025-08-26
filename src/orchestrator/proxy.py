import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from .llm_router import get_router
from .utils import (
    HeaderManager,
    SSEAccumulator,
    create_error_sse_message,
    extract_assistant_text,
)

# Configuration
load_dotenv()
CONFIG_FILE = Path("config.yaml")
config = yaml.safe_load(CONFIG_FILE.open("r", encoding="utf-8")) or {}
endpoints = config.get("endpoints", [])

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI()

# Global conversation store
convo_history: Dict[str, List[Dict]] = {}

# Global cache of reachable endpoints
reachable_endpoints: Dict[str, dict] = {}


class RequestProcessor:
    """Handles request processing, history management, and routing logic."""

    @staticmethod
    def get_endpoint_url(path: str, is_streaming: bool = False) -> Optional[str]:
        """Get target endpoint URL based on path and streaming preference."""
        endpoint_key = path.lstrip("/").split("/")[0] if path else ""

        for endpoint_config in endpoints:
            if endpoint_config.get("name") == endpoint_key:
                base_url = endpoint_config.get("url")
                if base_url:
                    suffix = (
                        "/v1/chat/completions"
                        if is_streaming
                        else "/api/v0/chat/completions"
                    )
                    return f"{base_url}{suffix}"

        logger.warning(f"No endpoint found for path: {path}")
        return None

    @staticmethod
    async def prepare_request(request: Request) -> Dict:
        """Parse and enrich request with conversation history and reasoning."""
        # Parse request
        try:
            raw_body = await request.body()
            body = json.loads(raw_body) if raw_body else {}
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON")

        if not isinstance(body, dict):
            return body

        # Get headers
        convo_id = request.headers.get("X-Convo-ID")
        reasoning_effort = request.headers.get("X-Reasoning-Effort", "").lower()
        reasoning_effort = (
            reasoning_effort if reasoning_effort in ["low", "medium", "high"] else None
        )

        # Handle conversation history
        messages = body.get("messages", [])
        if convo_id and messages:
            if convo_id not in convo_history:
                convo_history[convo_id] = []
            convo_history[convo_id].extend(messages)
            messages = convo_history[convo_id].copy()

        # Handle reasoning effort
        if messages:
            # Remove existing reasoning message
            if (
                messages
                and messages[0].get("role") == "system"
                and "Reasoning:" in messages[0].get("content", "")
            ):
                messages.pop(0)
                if convo_id:
                    convo_history[convo_id] = messages

            # Add new reasoning if provided
            if reasoning_effort:
                reasoning_msg = {
                    "role": "system",
                    "content": f"Reasoning: {reasoning_effort}",
                }
                messages.insert(0, reasoning_msg)
                if convo_id:
                    convo_history[convo_id] = messages
                logger.info(f"✅ Applied reasoning: {reasoning_effort}")

        # Update body
        body["messages"] = messages
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort

        return body

    @staticmethod
    def update_history(convo_id: Optional[str], assistant_text: str) -> None:
        """Update conversation history with assistant response."""
        if assistant_text and convo_id and convo_id in convo_history:
            convo_history[convo_id].append(
                {"role": "assistant", "content": assistant_text}
            )


class ProxyHandler:
    """Handles HTTP proxying to upstream endpoints."""

    @staticmethod
    async def stream_response(
        request: Request,
        target_url: str,
        body: Dict,
        convo_id: str,
        extra_headers: Dict = None,
    ) -> StreamingResponse:
        """Handle streaming response."""
        timeout = httpx.Timeout(connect=20, read=None, write=20, pool=20)
        headers = HeaderManager.prepare_upstream_headers(request, for_streaming=True)
        body["stream_options"] = {"include_usage": True}

        acc = SSEAccumulator()

        async def stream():
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream(
                        "POST", target_url, headers=headers, json=body
                    ) as resp:
                        resp.raise_for_status()
                        async for chunk in resp.aiter_bytes():
                            acc.feed(chunk)
                            yield chunk
            except httpx.HTTPStatusError as e:
                yield create_error_sse_message(
                    "error",
                    status=e.response.status_code,
                    detail=f"HTTP {e.response.status_code}",
                )
            except Exception as e:
                yield create_error_sse_message("error", detail=str(e))
            finally:
                try:
                    RequestProcessor.update_history(convo_id, acc.text())
                except Exception:
                    pass

        response_headers = HeaderManager.create_response_headers(
            convo_id=convo_id, for_streaming=True
        )
        if extra_headers:
            response_headers.update(extra_headers)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers=response_headers
        )

    @staticmethod
    async def non_stream_response(
        request: Request,
        target_url: str,
        body: Dict,
        convo_id: str,
        extra_headers: Dict = None,
    ) -> Response:
        """Handle non-streaming response."""
        headers = HeaderManager.prepare_upstream_headers(request)

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(target_url, headers=headers, json=body)

            # Extract assistant text for history
            assistant_text = None
            try:
                resp_json = resp.json()
                assistant_text = extract_assistant_text(resp_json)
                if "model" in resp_json:
                    logger.info(f"Response model: {resp_json['model']}")
            except Exception:
                logger.info(f"Response: {resp.text[-500:]}")

            RequestProcessor.update_history(convo_id, assistant_text)

            response_headers = HeaderManager.create_response_headers(
                dict(resp.headers), convo_id
            )
            if extra_headers:
                response_headers.update(extra_headers)

            return Response(resp.content, resp.status_code, response_headers)


##########  Core Proxy Function ##########
async def proxy_request(
    path: str, request: Request, extra_headers: Dict = None
) -> Response:
    """Main proxy handler - simplified and clean."""
    # Prepare request with history and reasoning
    body = await RequestProcessor.prepare_request(request)
    is_streaming = body.get("stream", False) or request.query_params.get("stream") in {
        "true",
        "1",
    }
    convo_id = request.headers.get("X-Convo-ID")

    # Get target endpoint
    target_url = RequestProcessor.get_endpoint_url(path, is_streaming)
    if not target_url:
        raise HTTPException(status_code=404, detail=f"Endpoint not found: {path}")

    logger.info(
        f"Proxying {request.method} {request.url.path} to {target_url} (streaming: {is_streaming})"
    )

    # Route to appropriate handler
    if is_streaming:
        return await ProxyHandler.stream_response(
            request, target_url, body, convo_id, extra_headers
        )
    else:
        return await ProxyHandler.non_stream_response(
            request, target_url, body, convo_id, extra_headers
        )


##########  API Endpoints ##########
@app.post("/")
async def root_chat(request: Request):
    """Route to first available endpoint."""
    if not endpoints:
        raise HTTPException(status_code=503, detail="No endpoints configured")

    first_endpoint = endpoints[0].get("name")
    if not first_endpoint:
        raise HTTPException(status_code=503, detail="Invalid endpoint configuration")

    logger.info(f"Routing root request to: {first_endpoint}")
    return await proxy_request(first_endpoint, request)


@app.post("/smart")
async def smart_route(request: Request):
    """Smart routing based on content analysis."""
    try:
        # Prepare request
        body = await request.json()
        messages = body.get("messages", [])

        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        # Get latest user message for routing
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user messages found")

        latest_message = user_messages[-1].get("content", "")

        # Route using LLM router
        router = get_router()
        decision = await router.route_request(latest_message, reachable_endpoints)
        endpoint_config = router.get_endpoint_by_name(decision.endpoint)

        # Build routing headers
        routing_headers = {
            "X-Route-Decision": decision.endpoint,
            "X-Route-Confidence": str(decision.confidence),
            "X-Route-Reason": decision.reason,
            "X-Route-Strategy": decision.workload_type.value,
        }

        # Add hardware info
        if endpoint_config:
            for attr in ["gpu", "vram", "soc", "cpu", "ram"]:
                value = getattr(endpoint_config, attr, None)
                if value:
                    routing_headers[f"X-Route-{attr.upper()}"] = value

        logger.info(
            f"Smart routing: {decision.endpoint} (confidence: {decision.confidence:.2f}) - {decision.reason}"
        )

        return await proxy_request(decision.endpoint, request, routing_headers)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Smart routing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Smart routing error: {str(e)}")


@app.post("/conversations/retrieve")
async def retrieve_conversation(request: Request):
    """Retrieve conversation history."""
    try:
        body = await request.json()
        convo_id = body.get("convo_id")

        if not convo_id:
            raise HTTPException(status_code=400, detail="Missing convo_id")

        if convo_id not in convo_history:
            raise HTTPException(
                status_code=404, detail=f"Conversation '{convo_id}' not found"
            )

        return convo_history[convo_id]

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except Exception as e:
        logger.error(f"Error retrieving conversation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/models")
async def list_models():
    """List all available models with metadata."""
    models = []

    for endpoint_config in endpoints:
        name = endpoint_config.get("name", "unknown")
        url = endpoint_config.get("url")

        if not url:
            logger.warning(f"No URL for endpoint: {name}")
            continue

        # Special handling for Auto endpoint
        if name == "Auto":
            models.append(
                {
                    "id": "auto-router",
                    "object": "model",
                    "created": int(datetime.now().timestamp()),
                    "owned_by": "llm-control-plane",
                    "endpoint": name,
                    "endpoint_url": url,
                    "description": "Intelligent routing to best available endpoint",
                }
            )
            continue

        # Fetch models from endpoint
        try:
            models_url = f"{url}/api/v0/models"
            headers = HeaderManager.create_auth_headers()

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(models_url, headers=headers)
                resp.raise_for_status()

                data = resp.json()
                remote_models = data.get(
                    "data", data.get("models", data if isinstance(data, list) else [])
                )

                for model in remote_models:
                    if isinstance(model, dict):
                        enhanced_model = {
                            "id": model.get("id", f"unknown-{name}"),
                            "object": "model",
                            "created": model.get(
                                "created", int(datetime.now().timestamp())
                            ),
                            "owned_by": model.get("owned_by", name),
                            "endpoint": name,
                            "endpoint_url": url,
                        }

                        # Add hardware specs
                        for hw in ["gpu", "vram", "soc", "cpu", "ram"]:
                            if hw in endpoint_config:
                                enhanced_model[hw] = endpoint_config[hw]

                        # Preserve original fields
                        for k, v in model.items():
                            if k not in enhanced_model:
                                enhanced_model[k] = v

                        models.append(enhanced_model)

        except httpx.HTTPError as e:
            logger.warning(f"Failed to fetch models from {name}: {e}")
        except Exception as e:
            logger.error(f"Error fetching models from {name}: {e}")

    global reachable_endpoints
    reachable_endpoints = list(
        set(model.get("endpoint") for model in models if model.get("endpoint"))
    )
    return {"object": "list", "data": models}


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def custom_endpoints(path: str, request: Request):
    """Handle all other endpoints."""
    return await proxy_request(path, request)
