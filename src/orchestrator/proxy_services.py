from __future__ import annotations

import asyncio
import logging
from typing import Optional

from dotenv import load_dotenv

from src.logging_config import configure_logging
from src.search import build_search_router

from .config import CONFIG_FILE, load_config
from .conversation_store import (
    ConversationStore,
    MemoryConversationStore,
    build_conversation_store_from_env,
)
from .orchestration import OrchestrationSubsystem
from .utils import HeaderManager

load_dotenv()
configure_logging()

config = load_config(CONFIG_FILE)
endpoints = config.get("endpoints", [])
retrieval_config = config.get("retrieval", {})
RETRIEVAL_TOP_K = int(retrieval_config.get("top_k", 3))
search_service = build_search_router(
    config.get("search", {}),
    query_refiner_headers=HeaderManager.create_auth_headers(),
)

logger = logging.getLogger(__name__)

conversation_store: ConversationStore = MemoryConversationStore()
workflow_registry = None
workflow_store = None
workflow_executor = None
graph_registry = None
graph_store = None
graph_executor = None
reachable_endpoints: list[str] = []
VALID_REASONING_EFFORTS = {"low", "medium", "high"}
conversation_finalization_tasks: set[asyncio.Task] = set()
stream_producer_tasks: set[asyncio.Task] = set()
SWITCH_WARNING_MESSAGE = (
    "Conversation endpoint/reasoning changed; full history was replayed to the "
    "selected endpoint."
)
orchestration_subsystems: list[OrchestrationSubsystem] = []


def _orchestration_config() -> dict:
    value = config.get("orchestration")
    return value if isinstance(value, dict) else {}


def _subsystem_enabled(name: str, *, default: bool) -> bool:
    section = _orchestration_config().get(name)
    if section is None:
        return default
    if isinstance(section, bool):
        return section
    if not isinstance(section, dict):
        return default
    enabled = section.get("enabled")
    if enabled is None:
        return default
    if isinstance(enabled, bool):
        return enabled
    return str(enabled).strip().lower() in {"1", "true", "yes", "on"}


def _build_orchestration_subsystems() -> list[OrchestrationSubsystem]:
    subsystems: list[OrchestrationSubsystem] = []
    if _subsystem_enabled("workflows", default=True):
        from .workflow.subsystem import WorkflowSubsystem

        subsystems.append(
            WorkflowSubsystem(
                conversation_store_getter=lambda: conversation_store,
            )
        )
    if _subsystem_enabled("graphs", default=True):
        from .graph.subsystem import GraphSubsystem

        subsystems.append(GraphSubsystem())
    return subsystems


def _find_subsystem(name: str) -> OrchestrationSubsystem | None:
    for subsystem in orchestration_subsystems:
        if subsystem.name == name:
            return subsystem
    return None


def _sync_workflow_globals() -> None:
    global workflow_registry, workflow_store, workflow_executor
    subsystem = _find_subsystem("workflows")
    workflow_registry = getattr(subsystem, "registry", None)
    workflow_store = getattr(subsystem, "store", None)
    workflow_executor = getattr(subsystem, "executor", None)


def _sync_graph_globals() -> None:
    global graph_registry, graph_store, graph_executor
    subsystem = _find_subsystem("graphs")
    graph_registry = getattr(subsystem, "registry", None)
    graph_store = getattr(subsystem, "store", None)
    graph_executor = getattr(subsystem, "executor", None)


orchestration_subsystems = _build_orchestration_subsystems()
_sync_workflow_globals()
_sync_graph_globals()


def track_conversation_finalization(task: asyncio.Task) -> asyncio.Task:
    conversation_finalization_tasks.add(task)

    def discard_done(done_task: asyncio.Task) -> None:
        conversation_finalization_tasks.discard(done_task)
        if done_task.cancelled():
            return
        exc = done_task.exception()
        if exc:
            logger.error(
                "Background conversation finalization failed",
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


async def drain_conversation_finalizations(timeout: float = 5.0) -> None:
    if not conversation_finalization_tasks:
        return
    try:
        await asyncio.wait_for(
            asyncio.gather(*conversation_finalization_tasks, return_exceptions=True),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Timed out waiting for %d conversation finalization task(s)",
            len(conversation_finalization_tasks),
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


def set_conversation_store(store: ConversationStore) -> None:
    global conversation_store
    conversation_store = store


def set_workflow_components(
    *,
    registry=None,
    store=None,
    executor=None,
) -> None:
    subsystem = _find_subsystem("workflows")
    if subsystem is None or not hasattr(subsystem, "set_components"):
        raise RuntimeError("Workflow subsystem is not enabled")
    subsystem.set_components(registry=registry, store=store, executor=executor)
    _sync_workflow_globals()


def set_graph_components(
    *,
    registry=None,
    store=None,
    executor=None,
) -> None:
    subsystem = _find_subsystem("graphs")
    if subsystem is None or not hasattr(subsystem, "set_components"):
        raise RuntimeError("Graph subsystem is not enabled")
    subsystem.set_components(registry=registry, store=store, executor=executor)
    _sync_graph_globals()


async def initialize_conversation_store() -> None:
    global conversation_store
    candidate = build_conversation_store_from_env()
    await candidate.initialize()
    conversation_store = candidate
    logger.info("Conversation history backend: %s", conversation_store.backend_name)


async def startup_conversation_store() -> None:
    await initialize_conversation_store()


async def shutdown_conversation_store() -> None:
    await drain_stream_producers()
    await drain_conversation_finalizations()
    await conversation_store.close()


def get_orchestration_subsystems() -> list[OrchestrationSubsystem]:
    return list(orchestration_subsystems)


async def startup_orchestration_subsystems(
    subsystems: list[OrchestrationSubsystem] | None = None,
) -> None:
    active_subsystems = subsystems if subsystems is not None else orchestration_subsystems
    started: list[OrchestrationSubsystem] = []
    try:
        for subsystem in active_subsystems:
            await subsystem.startup()
            started.append(subsystem)
            logger.info("Orchestration subsystem started: %s", subsystem.name)
        _sync_workflow_globals()
        _sync_graph_globals()
    except BaseException:
        for subsystem in reversed(started):
            await subsystem.shutdown()
        _sync_workflow_globals()
        _sync_graph_globals()
        raise


async def shutdown_orchestration_subsystems(
    subsystems: list[OrchestrationSubsystem] | None = None,
) -> None:
    active_subsystems = subsystems if subsystems is not None else orchestration_subsystems
    for subsystem in reversed(active_subsystems):
        await subsystem.shutdown()
        logger.info("Orchestration subsystem stopped: %s", subsystem.name)
    _sync_workflow_globals()
    _sync_graph_globals()


async def initialize_workflow_components() -> None:
    subsystem = _find_subsystem("workflows")
    if subsystem is None:
        raise RuntimeError("Workflow subsystem is not enabled")
    await subsystem.startup()
    _sync_workflow_globals()


async def startup_workflow_components() -> None:
    await initialize_workflow_components()


async def shutdown_workflow_components() -> None:
    subsystem = _find_subsystem("workflows")
    if subsystem is not None:
        await subsystem.shutdown()
    _sync_workflow_globals()


def get_workflow_registry():
    _sync_workflow_globals()
    if workflow_registry is None:
        raise RuntimeError("Workflow registry is not initialized")
    return workflow_registry


def get_workflow_store():
    _sync_workflow_globals()
    if workflow_store is None:
        raise RuntimeError("Workflow store is not initialized")
    return workflow_store


def get_workflow_executor():
    _sync_workflow_globals()
    if workflow_executor is None:
        raise RuntimeError("Workflow executor is not initialized")
    return workflow_executor


def get_graph_registry():
    _sync_graph_globals()
    if graph_registry is None:
        raise RuntimeError("Graph registry is not initialized")
    return graph_registry


def get_graph_store():
    _sync_graph_globals()
    if graph_store is None:
        raise RuntimeError("Graph store is not initialized")
    return graph_store


def get_graph_executor():
    _sync_graph_globals()
    if graph_executor is None:
        raise RuntimeError("Graph executor is not initialized")
    return graph_executor
