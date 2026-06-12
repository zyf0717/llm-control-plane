from __future__ import annotations

import asyncio
import logging
from typing import Optional

from dotenv import load_dotenv

from src.logging_config import configure_logging
from src.search import build_search_router

from .config import CONFIG_FILE, load_config
from .history_store import (
    HistoryStore,
    MemoryHistoryStore,
    build_history_store_from_env,
)
from .utils import HeaderManager
from .workflow import (
    SQLiteWorkflowStore,
    WorkflowExecutor,
    WorkflowRegistry,
    build_workflow_store_from_env,
)

load_dotenv()
configure_logging()

config = load_config(CONFIG_FILE)
endpoints = config.get("endpoints", [])
rag_config = config.get("rag", {})
RAG_TOP_K = int(rag_config.get("top_k", 3))
search_service = build_search_router(
    config.get("search", {}),
    query_refiner_headers=HeaderManager.create_auth_headers(),
)

logger = logging.getLogger(__name__)

history_store: HistoryStore = MemoryHistoryStore()
workflow_registry = WorkflowRegistry()
workflow_store: SQLiteWorkflowStore = build_workflow_store_from_env()
workflow_executor: Optional[WorkflowExecutor] = None
reachable_endpoints: list[str] = []
VALID_REASONING_EFFORTS = {"low", "medium", "high"}
history_finalization_tasks: set[asyncio.Task] = set()
stream_producer_tasks: set[asyncio.Task] = set()
SWITCH_WARNING_MESSAGE = (
    "Conversation endpoint/reasoning changed; full history was replayed to the "
    "selected endpoint."
)


def track_history_finalization(task: asyncio.Task) -> asyncio.Task:
    history_finalization_tasks.add(task)

    def discard_done(done_task: asyncio.Task) -> None:
        history_finalization_tasks.discard(done_task)
        if done_task.cancelled():
            return
        exc = done_task.exception()
        if exc:
            logger.error(
                "Background history finalization failed",
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    task.add_done_callback(discard_done)
    return task


def track_stream_producer(task: asyncio.Task) -> asyncio.Task:
    stream_producer_tasks.add(task)

    def discard_done(done_task: asyncio.Task) -> None:
        stream_producer_tasks.discard(done_task)
        if done_task.cancelled():
            return
        exc = done_task.exception()
        if exc:
            logger.error(
                "Background stream producer failed",
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    task.add_done_callback(discard_done)
    return task


async def drain_history_finalizations(timeout: float = 5.0) -> None:
    if not history_finalization_tasks:
        return
    try:
        await asyncio.wait_for(
            asyncio.gather(*history_finalization_tasks, return_exceptions=True),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Timed out waiting for %d history finalization task(s)",
            len(history_finalization_tasks),
        )


async def drain_stream_producers(timeout: float = 5.0) -> None:
    if not stream_producer_tasks:
        return
    try:
        await asyncio.wait_for(
            asyncio.gather(*stream_producer_tasks, return_exceptions=True),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Timed out waiting for %d stream producer task(s)",
            len(stream_producer_tasks),
        )


def set_history_store(store: HistoryStore) -> None:
    global history_store
    history_store = store


def set_workflow_components(
    *,
    registry: Optional[WorkflowRegistry] = None,
    store: Optional[SQLiteWorkflowStore] = None,
    executor: Optional[WorkflowExecutor] = None,
) -> None:
    global workflow_registry, workflow_store, workflow_executor
    if registry is not None:
        workflow_registry = registry
    if store is not None:
        workflow_store = store
    if executor is not None:
        workflow_executor = executor


async def initialize_history_store() -> None:
    global history_store
    candidate = build_history_store_from_env()
    await candidate.initialize()
    history_store = candidate
    logger.info("Conversation history backend: %s", history_store.backend_name)


async def startup_history_store() -> None:
    await initialize_history_store()


async def shutdown_history_store() -> None:
    await drain_stream_producers()
    await drain_history_finalizations()
    await history_store.close()


async def initialize_workflow_components() -> None:
    global workflow_registry, workflow_store, workflow_executor
    from .workflow_clients import ProxyWorkflowLLMClient, ProxyWorkflowSearchClient

    workflow_registry.load()
    workflow_store = build_workflow_store_from_env()
    await workflow_store.initialize()
    workflow_executor = WorkflowExecutor(
        workflow_registry,
        workflow_store,
        ProxyWorkflowLLMClient(),
        ProxyWorkflowSearchClient(),
    )


async def startup_workflow_components() -> None:
    await initialize_workflow_components()


async def shutdown_workflow_components() -> None:
    await workflow_store.close()


def get_workflow_registry() -> WorkflowRegistry:
    return workflow_registry


def get_workflow_store() -> SQLiteWorkflowStore:
    return workflow_store


def get_workflow_executor() -> WorkflowExecutor:
    if workflow_executor is None:
        raise RuntimeError("Workflow executor is not initialized")
    return workflow_executor
