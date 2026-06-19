from __future__ import annotations

import importlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from .models import GraphSpec


DEFAULT_LANGGRAPH_CONFIG = Path("langgraph.json")
DEFAULT_GRAPH_METADATA_DIR = Path("src/graphs")


class GraphRegistry:
    def __init__(
        self,
        config_path: Path | str = DEFAULT_LANGGRAPH_CONFIG,
        metadata_dir: Path | str = DEFAULT_GRAPH_METADATA_DIR,
    ):
        self.config_path = Path(config_path)
        self.metadata_dir = Path(metadata_dir)
        self._specs: dict[str, GraphSpec] = {}

    def load(self) -> None:
        if not self.config_path.exists():
            self._specs = {}
            return

        with self.config_path.open("r", encoding="utf-8") as handle:
            data: Any = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"graph config {self.config_path} must contain an object")

        graphs = data.get("graphs", {})
        if graphs is None:
            graphs = {}
        if not isinstance(graphs, dict):
            raise ValueError("langgraph.json graphs must be an object")

        specs: dict[str, GraphSpec] = {}
        for graph_id, graph_ref in sorted(graphs.items()):
            graph_id = _required_id(graph_id)
            if graph_id in specs:
                raise ValueError(f"duplicate graph id: {graph_id}")
            graph_ref = _required_ref(graph_ref, graph_id=graph_id)
            metadata = self._load_metadata(graph_id)
            graph = self._load_graph_ref(graph_ref, graph_id=graph_id)
            self._validate_graph_object(graph, graph_id=graph_id)
            specs[graph_id] = GraphSpec(
                id=graph_id,
                graph_ref=graph_ref,
                graph=graph,
                name=_optional_str(metadata.get("name")) or graph_id,
                description=_optional_str(metadata.get("description")),
                input_schema=_dict_or_empty(metadata.get("input_schema")),
                defaults=_dict_or_empty(metadata.get("defaults")),
                ui=_dict_or_empty(metadata.get("ui")),
            )

        self._specs = specs

    def list(self) -> list[GraphSpec]:
        return sorted(self._specs.values(), key=lambda spec: (spec.name, spec.id))

    def get(self, graph_id: str) -> GraphSpec:
        graph_id = str(graph_id or "").strip()
        try:
            return self._specs[graph_id]
        except KeyError as exc:
            raise KeyError(f"unknown graph: {graph_id}") from exc

    @property
    def loaded(self) -> bool:
        return bool(self._specs)

    def _load_metadata(self, graph_id: str) -> dict[str, Any]:
        path = self.metadata_dir / f"{graph_id}.yaml"
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                data: Any = yaml.safe_load(handle) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid graph metadata YAML in {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"graph metadata {path} must contain an object")
        metadata_id = _optional_str(data.get("id"))
        if metadata_id and metadata_id != graph_id:
            raise ValueError(
                f"graph metadata {path} id {metadata_id!r} does not match {graph_id!r}"
            )
        return data

    def _load_graph_ref(self, graph_ref: str, *, graph_id: str) -> Any:
        module_ref, _, attr_ref = graph_ref.partition(":")
        if not module_ref or not attr_ref:
            raise ValueError(
                f"graph {graph_id} ref must use '<module-or-path>:<attribute>'"
            )
        try:
            module = self._load_module(module_ref, graph_id=graph_id)
        except ModuleNotFoundError as exc:
            if exc.name == "langgraph":
                raise RuntimeError(
                    "LangGraph dependency is required for configured graphs; "
                    "install project dependencies in the llm-control-plane conda env"
                ) from exc
            raise
        current: Any = module
        for attr in attr_ref.split("."):
            attr = attr.strip()
            if not attr:
                raise ValueError(f"graph {graph_id} ref has an empty attribute segment")
            current = getattr(current, attr)
        return current

    def _load_module(self, module_ref: str, *, graph_id: str) -> Any:
        path = Path(module_ref)
        if module_ref.endswith(".py") or module_ref.startswith((".", "/")):
            module_path = path if path.is_absolute() else (Path.cwd() / path).resolve()
            if not module_path.exists():
                raise FileNotFoundError(
                    f"graph {graph_id} module path not found: {module_path}"
                )
            module_name = f"_llmcp_graph_{_safe_module_name(graph_id)}"
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load graph module: {module_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
        return importlib.import_module(module_ref)

    @staticmethod
    def _validate_graph_object(graph: Any, *, graph_id: str) -> None:
        if not any(hasattr(graph, name) for name in ("ainvoke", "invoke", "astream", "stream")):
            raise ValueError(
                f"graph {graph_id} object must expose invoke/ainvoke or stream/astream"
            )


def _required_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("graph id must be non-empty")
    return text


def _required_ref(value: Any, *, graph_id: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"graph {graph_id} ref must be non-empty")
    return text


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_module_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", value)
