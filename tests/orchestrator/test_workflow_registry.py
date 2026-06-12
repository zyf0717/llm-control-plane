from pathlib import Path

import pytest

from src.orchestrator.workflow.registry import DEFAULT_WORKFLOW_DIR, WorkflowRegistry


def write_workflow(path: Path, *, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def minimal_workflow(*, steps: str) -> str:
    return f"""
id: sample
name: Sample
version: 0.1.0
params_schema:
  type: object
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


def test_registry_loads_chat_visibility_fields(tmp_path):
    write_workflow(
        tmp_path / "sample.yaml",
        body=minimal_workflow(
            steps="""
  - id: first
    name: First
    kind: llm
    chat_visibility: intermediate
    chat_stream: false
    prompt: hello
"""
        ),
    )

    registry = WorkflowRegistry(tmp_path)
    registry.load()

    step = registry.get("sample").steps[0]
    assert step.chat_visibility == "intermediate"
    assert step.chat_stream is False


def test_registry_loads_search_control_fields(tmp_path):
    write_workflow(
        tmp_path / "sample.yaml",
        body=minimal_workflow(
            steps="""
  - id: search
    name: Search
    kind: search
    use_query_refiner: false
    use_reranker: true
    rerank_context: "Goal: {{ params.goal }}"
    prompt: hello
"""
        ),
    )

    registry = WorkflowRegistry(tmp_path)
    registry.load()

    step = registry.get("sample").steps[0]
    assert step.use_query_refiner is False
    assert step.use_reranker is True
    assert step.rerank_context == "Goal: {{ params.goal }}"


def test_registry_defaults_chat_visibility_to_hidden(tmp_path):
    write_workflow(
        tmp_path / "sample.yaml",
        body=minimal_workflow(
            steps="""
  - id: first
    name: First
    kind: manual
"""
        ),
    )

    registry = WorkflowRegistry(tmp_path)
    registry.load()

    step = registry.get("sample").steps[0]
    assert step.chat_visibility == "hidden"
    assert step.chat_stream is None


def test_registry_rejects_invalid_chat_visibility(tmp_path):
    write_workflow(
        tmp_path / "sample.yaml",
        body=minimal_workflow(
            steps="""
  - id: first
    name: First
    kind: manual
    chat_visibility: public
"""
        ),
    )

    with pytest.raises(ValueError, match="chat_visibility"):
        WorkflowRegistry(tmp_path).load()


def test_default_workflow_config_directory_loads_shipped_specs():
    assert DEFAULT_WORKFLOW_DIR.name == "workflow_configs"

    registry = WorkflowRegistry()
    registry.load()

    assert {spec.id for spec in registry.list()} == {
        "contextual_search",
        "implementation_plan",
        "research_brief",
    }


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
