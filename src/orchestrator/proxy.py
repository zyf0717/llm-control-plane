from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict

import httpx
from fastapi import FastAPI, HTTPException, Request

from . import proxy_services as services
from .llm_router import get_router
from .request_processor import RequestProcessor
from .search_routes import router as search_router
from .upstream_proxy import proxy_request
from .utils import HeaderManager
from .workflow import create_workflow_router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await services.startup_history_store()
    await services.startup_workflow_components()
    try:
        yield
    finally:
        await services.shutdown_workflow_components()
        await services.shutdown_history_store()


app = FastAPI(lifespan=lifespan)

app.include_router(
    create_workflow_router(
        registry_getter=services.get_workflow_registry,
        store_getter=services.get_workflow_store,
        executor_getter=services.get_workflow_executor,
    )
)
app.include_router(search_router)


@app.post("/")
async def root_chat(request: Request):
    """Route to smart routing endpoint."""
    services.logger.info("Routing root request to smart routing")
    return await smart_route(request)


@app.post("/smart")
@app.post("/smart/{subpath:path}")
async def smart_route(request: Request, subpath: str = ""):
    """Smart routing based on content analysis."""
    try:
        body = await request.json()
        messages = body.get("messages", [])

        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user messages found")

        convo_id = RequestProcessor._normalize_convo_id(
            request.headers.get("X-Convo-ID")
        )
        valid_endpoints = RequestProcessor.configured_endpoint_names()
        stale_headers: Dict[str, str] = {}
        if convo_id:
            state = await services.history_store.get_conversation_state(convo_id)
            pinned_endpoint = str(state.get("route_endpoint") or "").strip()
            if pinned_endpoint and pinned_endpoint in valid_endpoints:
                routing_headers = {
                    "X-Route-Decision": pinned_endpoint,
                    "X-Route-Pinned": "true",
                    "X-Route-Pin-Stale": "false",
                }
                return await proxy_request(
                    pinned_endpoint,
                    request,
                    routing_headers,
                    route_conflict_policy="use-existing",
                )
            if pinned_endpoint and pinned_endpoint not in valid_endpoints:
                await services.history_store.update_conversation_state(
                    convo_id,
                    valid_route_endpoints=valid_endpoints,
                    clear_route=True,
                )
                stale_headers["X-Route-Pin-Stale"] = "true"

        latest_message = user_messages[-1].get("content", "")

        router = get_router()
        decision = await router.route_request(
            latest_message,
            services.reachable_endpoints,
        )
        endpoint_config = router.get_endpoint_by_name(decision.endpoint)

        routing_headers = {
            "X-Route-Decision": decision.endpoint,
            "X-Route-Confidence": str(decision.confidence),
            "X-Route-Reason": decision.reason,
            "X-Route-Strategy": decision.workload_type.value,
            "X-Route-Pinned": "false",
            "X-Route-Pin-Stale": stale_headers.get("X-Route-Pin-Stale", "false"),
        }

        if endpoint_config:
            for attr in ["gpu", "vram", "soc", "cpu", "ram"]:
                value = getattr(endpoint_config, attr, None)
                if value:
                    routing_headers[f"X-Route-{attr.upper()}"] = value

        services.logger.info(
            "Smart routing: %s (confidence: %.2f) - %s",
            decision.endpoint,
            decision.confidence,
            decision.reason,
        )

        return await proxy_request(
            decision.endpoint,
            request,
            routing_headers,
            route_conflict_policy="use-existing",
        )

    except HTTPException:
        raise
    except Exception as exc:
        services.logger.error("Smart routing failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Smart routing error: {str(exc)}",
        ) from exc


@app.post("/conversations/retrieve")
async def retrieve_conversation(request: Request):
    """Retrieve conversation history."""
    try:
        body = await request.json()
        convo_id = body.get("convo_id")

        if not convo_id:
            raise HTTPException(status_code=400, detail="Missing convo_id")

        conversation = await services.history_store.get_conversation(convo_id)
        if conversation is None:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation '{convo_id}' not found",
            )

        return RequestProcessor._filter_ephemeral_search_messages(conversation)

    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc
    except HTTPException:
        raise
    except Exception as exc:
        services.logger.error("Error retrieving conversation: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@app.post("/conversations/state")
async def retrieve_conversation_state(request: Request):
    """Retrieve route/reasoning/slot state for a conversation."""
    try:
        body = await request.json()
        convo_id = body.get("convo_id")

        if not convo_id:
            raise HTTPException(status_code=400, detail="Missing convo_id")

        return await services.history_store.get_conversation_state(str(convo_id))

    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc
    except HTTPException:
        raise
    except Exception as exc:
        services.logger.error("Error retrieving conversation state: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@app.get("/conversations")
async def list_conversations():
    """List conversation metadata sorted by most recent activity."""
    try:
        return await services.history_store.list_conversations()
    except Exception as exc:
        services.logger.error("Error listing conversations: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@app.get("/models")
async def list_models():
    """List all available models with metadata."""
    models = []

    async def fetch_endpoint_models(endpoint_config):
        """Fetch models from a single endpoint."""
        name = endpoint_config.get("name", "unknown")
        url = endpoint_config.get("url")

        if not url:
            services.logger.warning("No URL for endpoint: %s", name)
            return []

        try:
            headers = HeaderManager.create_auth_headers()

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = None
                for models_url in [f"{url}/v1/models"]:
                    try:
                        resp = await client.get(models_url, headers=headers)
                        resp.raise_for_status()
                        services.logger.debug(
                            "Successfully fetched models from %s",
                            models_url,
                        )
                        break
                    except httpx.HTTPError as exc:
                        services.logger.debug(
                            "Failed to fetch from %s: %s",
                            models_url,
                            exc,
                        )
                        continue

                if resp is None:
                    services.logger.warning(
                        "Failed to fetch models from %s using all endpoints",
                        name,
                    )
                    return []

                data = resp.json()
                remote_models = data.get(
                    "data",
                    data.get("models", data if isinstance(data, list) else []),
                )

                endpoint_models = []
                for model in remote_models:
                    if isinstance(model, dict):
                        enhanced_model = {
                            "id": model.get("id", f"unknown-{name}"),
                            "object": "model",
                            "created": model.get(
                                "created",
                                int(datetime.now().timestamp()),
                            ),
                            "owned_by": model.get("owned_by", name),
                            "endpoint": name,
                            "endpoint_url": url,
                        }

                        for hw in ["gpu", "vram", "soc", "cpu", "ram"]:
                            if hw in endpoint_config:
                                enhanced_model[hw] = endpoint_config[hw]

                        for key, value in model.items():
                            if key not in enhanced_model:
                                enhanced_model[key] = value

                        endpoint_models.append(enhanced_model)

                return endpoint_models

        except httpx.HTTPError as exc:
            services.logger.warning("Failed to fetch models from %s: %s", name, exc)
            return []
        except Exception as exc:
            services.logger.error("Error fetching models from %s: %s", name, exc)
            return []

    tasks = [
        fetch_endpoint_models(endpoint_config)
        for endpoint_config in services.endpoints
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, list):
            models.extend(result)
        elif isinstance(result, Exception):
            services.logger.error("Endpoint fetch failed: %s", result)

    services.reachable_endpoints = list(
        set(model.get("endpoint") for model in models if model.get("endpoint"))
    )
    return {"object": "list", "data": models}


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def custom_endpoints(path: str, request: Request):
    """Handle all other endpoints."""
    return await proxy_request(path, request)
