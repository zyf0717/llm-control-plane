from __future__ import annotations

from typing import Callable

from fastapi import APIRouter

from ..conversation_store import ConversationStore
from ..runtime import ProxyRuntimeLLMClient, ProxyRuntimeSearchClient
from .api import create_workflow_router
from .executor import WorkflowExecutor
from .registry import WorkflowRegistry
from .store import SQLiteWorkflowStore, build_workflow_store_from_env


class WorkflowSubsystem:
    name = "workflows"

    def __init__(
        self,
        *,
        conversation_store_getter: Callable[[], ConversationStore] | None = None,
    ):
        self.conversation_store_getter = conversation_store_getter
        self.registry = WorkflowRegistry()
        self.store: SQLiteWorkflowStore = build_workflow_store_from_env()
        self.executor: WorkflowExecutor | None = None

    def router(self) -> APIRouter:
        return create_workflow_router(
            registry_getter=lambda: self.registry,
            store_getter=lambda: self.store,
            executor_getter=self.get_executor,
            conversation_store_getter=self.conversation_store_getter,
        )

    async def startup(self) -> None:
        self.registry.load()
        self.store = build_workflow_store_from_env()
        await self.store.initialize()
        self.executor = WorkflowExecutor(
            self.registry,
            self.store,
            ProxyRuntimeLLMClient(),
            ProxyRuntimeSearchClient(),
        )

    async def shutdown(self) -> None:
        await self.store.close()

    def health(self) -> dict[str, object]:
        return {
            "name": self.name,
            "loaded": self.registry.loaded,
            "backend": self.store.backend_name,
            "executor_initialized": self.executor is not None,
        }

    def get_executor(self) -> WorkflowExecutor:
        if self.executor is None:
            raise RuntimeError("Workflow executor is not initialized")
        return self.executor

    def set_components(
        self,
        *,
        registry: WorkflowRegistry | None = None,
        store: SQLiteWorkflowStore | None = None,
        executor: WorkflowExecutor | None = None,
    ) -> None:
        if registry is not None:
            self.registry = registry
        if store is not None:
            self.store = store
        if executor is not None:
            self.executor = executor

