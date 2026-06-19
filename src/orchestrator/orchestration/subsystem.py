from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from fastapi import APIRouter


@dataclass(frozen=True, slots=True)
class DashboardDescriptor:
    tab_id: str
    label: str
    client_base_path: str


class OrchestrationSubsystem(Protocol):
    name: str

    def router(self) -> APIRouter:
        ...

    async def startup(self) -> None:
        ...

    async def shutdown(self) -> None:
        ...

    def health(self) -> dict[str, object]:
        ...

