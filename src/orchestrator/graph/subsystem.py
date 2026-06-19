from __future__ import annotations

from fastapi import APIRouter

from .api import create_graph_router
from .executor import GraphExecutor
from .registry import GraphRegistry
from .store import SQLiteGraphRunStore, build_graph_store_from_env


class GraphSubsystem:
    name = "graphs"

    def __init__(self):
        self.registry = GraphRegistry()
        self.store: SQLiteGraphRunStore = build_graph_store_from_env()
        self.executor: GraphExecutor | None = None

    def router(self) -> APIRouter:
        return create_graph_router(
            registry_getter=lambda: self.registry,
            store_getter=lambda: self.store,
            executor_getter=self.get_executor,
        )

    async def startup(self) -> None:
        self.registry.load()
        self.store = build_graph_store_from_env()
        await self.store.initialize()
        self.executor = GraphExecutor(self.registry, self.store)

    async def shutdown(self) -> None:
        await self.store.close()

    def health(self) -> dict[str, object]:
        return {
            "name": self.name,
            "loaded": self.registry.loaded,
            "backend": self.store.backend_name,
            "executor_initialized": self.executor is not None,
        }

    def get_executor(self) -> GraphExecutor:
        if self.executor is None:
            raise RuntimeError("Graph executor is not initialized")
        return self.executor

    def set_components(
        self,
        *,
        registry: GraphRegistry | None = None,
        store: SQLiteGraphRunStore | None = None,
        executor: GraphExecutor | None = None,
    ) -> None:
        if registry is not None:
            self.registry = registry
        if store is not None:
            self.store = store
        if executor is not None:
            self.executor = executor

