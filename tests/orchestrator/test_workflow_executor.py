from pathlib import Path

import pytest

from src.orchestrator.workflow_executor import WorkflowExecutor, render_template
from src.orchestrator.workflow_registry import WorkflowRegistry
from src.orchestrator.workflow_store import SQLiteWorkflowStore


class FakeLLMClient:
    def __init__(self):
        self.prompts = []

    async def complete(self, **kwargs):
        self.prompts.append(kwargs)
        return {
            "text": f"answer {len(self.prompts)}: {kwargs['prompt']}",
            "metadata": {"endpoint": kwargs["endpoint"]},
        }


def write_workflow(path: Path) -> None:
    path.write_text(
        """
id: sample
name: Sample
version: 0.1.0
params_schema:
  type: object
  required: [goal]
defaults:
  endpoint: smart
  reasoning_effort: high
steps:
  - id: first
    kind: llm
    prompt: "Goal: {{ params.goal }}"
    output_key: first_out
  - id: second
    kind: llm
    depends_on: [first]
    prompt: "Previous: {{ outputs.first_out.text }}"
    output_key: second_out
""",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_executor_advances_steps_with_previous_outputs(tmp_path):
    write_workflow(tmp_path / "sample.yaml")
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    llm = FakeLLMClient()
    executor = WorkflowExecutor(registry, store, llm)
    try:
        created = await executor.create_run("sample", params={"goal": "ship"})
        run_id = created["run"]["run_id"]

        first = await executor.advance(run_id)
        second = await executor.advance(run_id)

        assert first["steps"][0]["status"] == "completed"
        assert second["run"]["status"] == "completed"
        assert "Goal: ship" in llm.prompts[0]["prompt"]
        assert "answer 1" in llm.prompts[1]["prompt"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_executor_stores_failure_and_stops(tmp_path):
    class FailingLLMClient:
        async def complete(self, **kwargs):
            raise RuntimeError("upstream failed")

    write_workflow(tmp_path / "sample.yaml")
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    executor = WorkflowExecutor(registry, store, FailingLLMClient())
    try:
        created = await executor.create_run("sample", params={"goal": "ship"})
        snapshot = await executor.advance(created["run"]["run_id"])
        retried_without_retry = await executor.advance(created["run"]["run_id"])

        assert snapshot["run"]["status"] == "failed"
        assert snapshot["steps"][0]["status"] == "failed"
        assert snapshot["steps"][0]["error"] == "upstream failed"
        assert snapshot["steps"][1]["status"] == "pending"
        assert retried_without_retry["run"]["status"] == "failed"
        assert retried_without_retry["steps"][0]["status"] == "failed"
    finally:
        await store.close()


def test_render_template_resolves_nested_values_without_execution():
    rendered = render_template(
        "{{ params.goal }} / {{ outputs.step.text }} / {{ missing.value }}",
        {"params": {"goal": "ship"}, "outputs": {"step": {"text": "done"}}},
    )

    assert rendered == "ship / done / "
