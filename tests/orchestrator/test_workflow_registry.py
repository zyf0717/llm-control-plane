from pathlib import Path

import pytest

from src.orchestrator.workflow_registry import WorkflowRegistry


def write_workflow(path: Path, *, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def minimal_workflow(*, steps: str) -> str:
    return f"""
id: sample
name: Sample
version: 0.1.0
params_schema:
  type: object
defaults:
  endpoint: smart
steps:
{steps}
"""


def test_registry_loads_valid_specs(tmp_path):
    write_workflow(
        tmp_path / "sample.yaml",
        body=minimal_workflow(
            steps="""
  - id: first
    name: First
    kind: manual
  - id: second
    name: Second
    kind: manual
    depends_on: [first]
"""
        ),
    )

    registry = WorkflowRegistry(tmp_path)
    registry.load()

    assert [spec.id for spec in registry.list()] == ["sample"]
    assert registry.get("sample").steps[1].depends_on == ["first"]


def test_registry_rejects_duplicate_workflow_ids(tmp_path):
    body = minimal_workflow(
        steps="""
  - id: first
    kind: manual
"""
    )
    write_workflow(tmp_path / "one.yaml", body=body)
    write_workflow(tmp_path / "two.yaml", body=body)

    with pytest.raises(ValueError, match="duplicate workflow id"):
        WorkflowRegistry(tmp_path).load()


def test_registry_rejects_unknown_dependencies(tmp_path):
    write_workflow(
        tmp_path / "sample.yaml",
        body=minimal_workflow(
            steps="""
  - id: first
    kind: manual
    depends_on: [missing]
"""
        ),
    )

    with pytest.raises(ValueError, match="unknown step"):
        WorkflowRegistry(tmp_path).load()


def test_registry_rejects_cycles(tmp_path):
    write_workflow(
        tmp_path / "sample.yaml",
        body=minimal_workflow(
            steps="""
  - id: first
    kind: manual
    depends_on: [second]
  - id: second
    kind: manual
    depends_on: [first]
"""
        ),
    )

    with pytest.raises(ValueError, match="cyclic"):
        WorkflowRegistry(tmp_path).load()
