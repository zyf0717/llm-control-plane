from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .workflow_models import WorkflowSpec


DEFAULT_WORKFLOW_DIR = Path("workflows")


class WorkflowRegistry:
    def __init__(self, workflow_dir: Path | str = DEFAULT_WORKFLOW_DIR):
        self.workflow_dir = Path(workflow_dir)
        self._specs: dict[str, WorkflowSpec] = {}

    def load(self) -> None:
        specs: dict[str, WorkflowSpec] = {}
        if not self.workflow_dir.exists():
            raise FileNotFoundError(f"workflow directory not found: {self.workflow_dir}")

        for path in sorted(self.workflow_dir.glob("*.yaml")):
            spec = self._load_file(path)
            if spec.id in specs:
                raise ValueError(f"duplicate workflow id: {spec.id}")
            self._validate_spec(spec)
            specs[spec.id] = spec

        self._specs = specs

    def list(self) -> list[WorkflowSpec]:
        return sorted(self._specs.values(), key=lambda spec: (spec.name, spec.id))

    def get(self, workflow_id: str) -> WorkflowSpec:
        workflow_id = str(workflow_id or "").strip()
        try:
            return self._specs[workflow_id]
        except KeyError as exc:
            raise KeyError(f"unknown workflow: {workflow_id}") from exc

    @property
    def loaded(self) -> bool:
        return bool(self._specs)

    @staticmethod
    def _load_file(path: Path) -> WorkflowSpec:
        try:
            with path.open("r", encoding="utf-8") as handle:
                data: Any = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid workflow YAML in {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"workflow file {path} must contain an object")
        try:
            return WorkflowSpec.from_dict(data)
        except ValueError as exc:
            raise ValueError(f"invalid workflow spec {path}: {exc}") from exc

    @staticmethod
    def _validate_spec(spec: WorkflowSpec) -> None:
        step_ids: set[str] = set()
        for step in spec.steps:
            if step.id in step_ids:
                raise ValueError(f"workflow {spec.id} has duplicate step id: {step.id}")
            step_ids.add(step.id)

        for step in spec.steps:
            for dependency in step.depends_on or []:
                if dependency not in step_ids:
                    raise ValueError(
                        f"workflow {spec.id} step {step.id} depends on unknown step: "
                        f"{dependency}"
                    )

        WorkflowRegistry._reject_cycles(spec.id, {step.id: step.depends_on or [] for step in spec.steps})

    @staticmethod
    def _reject_cycles(workflow_id: str, graph: dict[str, list[str]]) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(step_id: str) -> None:
            if step_id in visited:
                return
            if step_id in visiting:
                raise ValueError(f"workflow {workflow_id} has cyclic step dependencies")
            visiting.add(step_id)
            for dependency in graph.get(step_id, []):
                visit(dependency)
            visiting.remove(step_id)
            visited.add(step_id)

        for step_id in graph:
            visit(step_id)
