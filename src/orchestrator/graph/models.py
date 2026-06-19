from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class GraphSpec:
    id: str
    graph_ref: str
    graph: Any
    name: str
    description: str | None = None
    input_schema: dict[str, Any] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=dict)
    ui: dict[str, Any] = field(default_factory=dict)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "graph_ref": self.graph_ref,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.summary_dict(),
            "input_schema": dict(self.input_schema),
            "defaults": dict(self.defaults),
            "ui": dict(self.ui),
        }

