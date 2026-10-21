from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from . import proxy_services as services
from .thread_state_refresh import maybe_refresh_thread_state
from .request_processor import RequestProcessor
from .repo_context_routes import router as repo_context_router
from .search_routes import router as search_router
from .upstream_proxy import proxy_request


def create_app(*, orchestration_subsystems=None) -> FastAPI:
    subsystems = (
        list(orchestration_subsystems)
        if orchestration_subsystems is not None
        else services.get_orchestration_subsystems()
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await services.startup_conversation_store()
        await services.startup_orchestration_subsystems(subsystems)
        try:
            yield
        finally:
            await services.shutdown_orchestration_subsystems(subsystems)
            await services.shutdown_conversation_store()

    created = FastAPI(lifespan=lifespan)
    for subsystem in subsystems:
        created.include_router(subsystem.router())
    created.include_router(search_router)
    created.include_router(repo_context_router)
    return created


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await services.startup_conversation_store()
    await services.startup_orchestration_subsystems()
    try:
        yield
    finally:
        await services.shutdown_orchestration_subsystems()
        await services.shutdown_conversation_store()


app = FastAPI(lifespan=lifespan)
for _subsystem in services.get_orchestration_subsystems():
    app.include_router(_subsystem.router())
app.include_router(search_router)
app.include_router(repo_context_router)


@app.post("/")
async def root_chat(request: Request):
    """Route to smart routing endpoint."""
    services.logger.info("Routing root request to smart routing")
    return await smart_route(request)


@app.post("/smart")
@app.post("/smart/{subpath:path}")
async def smart_route(request: Request, subpath: str = ""):
    return await proxy_request("smart", request)


@app.post("/conversations/retrieve")
async def retrieve_conversation(request: Request):
    """Retrieve conversation history."""
    try:
        body = await request.json()
        conversation_id = body.get("conversation_id")

        if not conversation_id:
            raise HTTPException(status_code=400, detail="Missing conversation_id")

        conversation = await services.conversation_store.get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation '{conversation_id}' not found",
            )

        return RequestProcessor.public_conversation_history(conversation)

    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc
    except HTTPException:
        raise
    except Exception as exc:
        services.logger.error("Error retrieving conversation: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@app.post("/conversations/append")
async def append_conversation(request: Request):
    """Append dashboard-owned public transcript messages."""
    try:
        body = await request.json()
        conversation_id = body.get("conversation_id")
        messages = body.get("messages")

        if not conversation_id:
            raise HTTPException(status_code=400, detail="Missing conversation_id")
        if not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="Missing messages")

        public_messages = RequestProcessor.public_conversation_history(messages)
        if not public_messages:
            raise HTTPException(status_code=400, detail="No public messages to append")

        await services.conversation_store.append_messages(
            str(conversation_id), public_messages
        )
        return {"appended": len(public_messages)}

    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc
    except HTTPException:
        raise
    except Exception as exc:
        services.logger.error("Error appending conversation: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@app.post("/conversations/state")
async def retrieve_conversation_control_state(request: Request):
    """Retrieve route/reasoning/slot state for a conversation."""
    try:
        body = await request.json()
        conversation_id = body.get("conversation_id")

        if not conversation_id:
            raise HTTPException(status_code=400, detail="Missing conversation_id")

        return await services.conversation_store.get_conversation_control_state(str(conversation_id))

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
        return await services.conversation_store.list_conversations()
    except Exception as exc:
        services.logger.error("Error listing conversations: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@app.get("/threads/{conversation_id}/state")
async def get_thread_state(conversation_id: str):
    """Retrieve derived thread state metadata."""
    state = await services.conversation_store.get_thread_state(conversation_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail=f"Thread state for conversation '{conversation_id}' not found",
        )
    return state


@app.post("/threads/{conversation_id}/refresh")
async def refresh_thread_state(conversation_id: str, request: Request):
    """Refresh derived thread state on demand."""
    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Invalid JSON object")

    endpoint = str(body.get("endpoint") or "").strip()
    if not endpoint:
        raise HTTPException(status_code=400, detail="Missing endpoint")

    try:
        llm_client = services.get_workflow_executor().llm_client
    except RuntimeError:
        from .runtime import ProxyRuntimeLLMClient

        llm_client = ProxyRuntimeLLMClient()

    try:
        return await maybe_refresh_thread_state(
            conversation_store=services.conversation_store,
            llm_client=llm_client,
            source_conversation_id=conversation_id,
            endpoint=endpoint,
            reasoning_effort=RequestProcessor._normalize_reasoning_effort(
                body.get("reasoning_effort")
            ),
            force=_truthy_body_value(body.get("force")),
        )
    except Exception as exc:
        services.logger.error("Error refreshing thread state: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


def _truthy_body_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


@app.get("/models")
async def list_models():
    return await services.llm_router.list_models()


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def custom_endpoints(path: str, request: Request):
    """Handle all other endpoints."""
    return await proxy_request(path, request)
