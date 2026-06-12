from pathlib import Path

import pytest

from src.orchestrator.workflow.executor import WorkflowExecutor, render_template
from src.orchestrator.workflow.registry import WorkflowRegistry
from src.orchestrator.workflow.store import SQLiteWorkflowStore


class FakeLLMClient:
    def __init__(self):
        self.prompts = []

    async def complete(self, **kwargs):
        self.prompts.append(kwargs)
        return {
            "text": f"answer {len(self.prompts)}: {kwargs['prompt']}",
            "metadata": {"endpoint": kwargs["endpoint"]},
        }


class JsonLLMClient:
    def __init__(self, text: str):
        self.text = text
        self.prompts = []

    async def complete(self, **kwargs):
        self.prompts.append(kwargs)
        return {"text": self.text, "metadata": {"endpoint": kwargs["endpoint"]}}


class BlockingLLMClient:
    def __init__(self):
        self.started = False
        self.release = None

    async def complete(self, **kwargs):
        import asyncio

        self.started = True
        self.release = asyncio.Event()
        await self.release.wait()
        return {"text": "released", "metadata": {"endpoint": kwargs["endpoint"]}}


class CapturingSearchClient:
    def __init__(self):
        self.calls = []

    async def search(self, **kwargs):
        self.calls.append(kwargs)
        return {"results": [{"title": "Result", "url": "https://example.com"}]}


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


def write_context_workflow(path: Path) -> None:
    path.write_text(
        """
id: context_sample
name: Context Sample
version: 0.1.0
params_schema:
  type: object
  required: [goal]
defaults:
  reasoning_effort: high
  search_provider: wikipedia_opensearch
steps:
  - id: search_context
    kind: search
    prompt: "{{ params.goal }}"
    output_key: search_context
  - id: synthesize
    kind: llm
    depends_on: [search_context]
    prompt: "{{ outputs.search_context.text }}"
    output_key: synthesis
""",
        encoding="utf-8",
    )


def write_json_query_workflow(path: Path) -> None:
    path.write_text(
        """
id: json_query_sample
name: JSON Query Sample
version: 0.1.0
params_schema:
  type: object
  required: [goal]
steps:
  - id: plan
    kind: llm
    prompt: "{{ params.goal }}"
    output_key: plan
  - id: search
    kind: search
    depends_on: [plan]
    prompt: "{{ outputs.plan.json.query }}"
    output_key: search
""",
        encoding="utf-8",
    )


def write_json_queries_workflow(path: Path) -> None:
    path.write_text(
        """
id: json_queries_sample
name: JSON Queries Sample
version: 0.1.0
params_schema:
  type: object
  required: [goal]
steps:
  - id: plan
    kind: llm
    prompt: "{{ params.goal }}"
    output_key: plan
  - id: search
    kind: search
    depends_on: [plan]
    prompt: "{{ outputs.plan.json.queries }}"
    output_key: search
""",
        encoding="utf-8",
    )


def write_direct_search_workflow(path: Path) -> None:
    path.write_text(
        """
id: direct_search_sample
name: Direct Search Sample
version: 0.1.0
params_schema:
  type: object
  required: [goal]
steps:
  - id: search
    kind: search
    prompt: "{{ json params.goal }}"
    output_key: search
  - id: synthesize
    kind: llm
    depends_on: [search]
    prompt: "{{ outputs.search.text }}"
    output_key: synthesis
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
        created = await executor.create_run(
            "sample", params={"goal": "ship"}, endpoint="node-a"
        )
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
async def test_executor_uses_run_search_provider_and_rag_endpoint(tmp_path):
    write_context_workflow(tmp_path / "context_sample.yaml")
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    llm = FakeLLMClient()
    search = CapturingSearchClient()
    executor = WorkflowExecutor(registry, store, llm, search)
    try:
        created = await executor.create_run(
            "context_sample",
            params={"goal": "ship"},
            endpoint="node-a",
            rag_endpoint="http://rag/api/retrieve/context",
            search_provider="duckduckgo_html",
        )
        run_id = created["run"]["run_id"]

        await executor.advance(run_id)
        await executor.advance(run_id)

        assert search.calls[0]["provider"] == "duckduckgo_html"
        assert llm.prompts[0]["rag_endpoint"] == "http://rag/api/retrieve/context"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_search_step_can_use_json_field_from_previous_llm_output(tmp_path):
    write_json_query_workflow(tmp_path / "json_query_sample.yaml")
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    llm = JsonLLMClient(
        '{"query": "site:platform.openai.com/docs Responses API function calling"}'
    )
    search = CapturingSearchClient()
    executor = WorkflowExecutor(registry, store, llm, search)
    try:
        created = await executor.create_run(
            "json_query_sample", params={"goal": "ship"}, endpoint="node-a"
        )
        run_id = created["run"]["run_id"]

        await executor.advance(run_id)
        await executor.advance(run_id)

        assert search.calls[0]["query"] == (
            "site:platform.openai.com/docs Responses API function calling"
        )
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_search_step_fans_out_workflow_planned_queries_without_planner(tmp_path):
    write_json_queries_workflow(tmp_path / "json_queries_sample.yaml")
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    llm = JsonLLMClient('{"queries": ["query one", "query two", "query one"]}')
    search = CapturingSearchClient()
    executor = WorkflowExecutor(registry, store, llm, search)
    try:
        created = await executor.create_run(
            "json_queries_sample", params={"goal": "ship"}, endpoint="node-a"
        )
        run_id = created["run"]["run_id"]

        await executor.advance(run_id)
        snapshot = await executor.advance(run_id)

        assert [call["query"] for call in search.calls] == ["query one", "query two"]
        assert [call["use_planner"] for call in search.calls] == [False, False]
        search_output = snapshot["steps"][1]["output_json"]["json"]
        assert search_output["queries"] == ["query one", "query two"]
        assert search_output["workflow_search"]["planned_by_workflow"] is True
        assert [item["query"] for item in search_output["per_query"]] == [
            "query one",
            "query two",
        ]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_llm_json_output_parser_accepts_fenced_json_for_workflow_queries(tmp_path):
    write_json_queries_workflow(tmp_path / "json_queries_sample.yaml")
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    llm = JsonLLMClient(
        '```json\n{"queries": ["query one", "query two"]}\n```'
    )
    search = CapturingSearchClient()
    executor = WorkflowExecutor(registry, store, llm, search)
    try:
        created = await executor.create_run(
            "json_queries_sample", params={"goal": "ship"}, endpoint="node-a"
        )
        run_id = created["run"]["run_id"]

        await executor.advance(run_id)
        snapshot = await executor.advance(run_id)

        assert [call["query"] for call in search.calls] == ["query one", "query two"]
        assert snapshot["steps"][0]["output_json"]["json"] == {
            "queries": ["query one", "query two"]
        }
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_search_step_can_dispatch_json_string_query_without_planner(tmp_path):
    write_direct_search_workflow(tmp_path / "direct_search_sample.yaml")
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    llm = FakeLLMClient()
    search = CapturingSearchClient()
    executor = WorkflowExecutor(registry, store, llm, search)
    try:
        created = await executor.create_run(
            "direct_search_sample",
            params={"goal": 'best "portable induction" cooktop'},
            endpoint="node-a",
        )
        run_id = created["run"]["run_id"]

        await executor.advance(run_id)

        assert search.calls == [
            {
                "query": 'best "portable induction" cooktop',
                "provider": None,
                "use_planner": False,
            }
        ]
        assert llm.prompts == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_executor_requires_concrete_run_endpoint(tmp_path):
    write_workflow(tmp_path / "sample.yaml")
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    executor = WorkflowExecutor(registry, store, FakeLLMClient())
    try:
        with pytest.raises(ValueError, match="endpoint is required"):
            await executor.create_run("sample", params={"goal": "ship"})

        with pytest.raises(ValueError, match="concrete endpoint"):
            await executor.create_run(
                "sample", params={"goal": "ship"}, endpoint="smart"
            )
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_executor_retry_completed_step_resets_downstream_outputs_and_artifacts(
    tmp_path,
):
    write_workflow(tmp_path / "sample.yaml")
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    llm = FakeLLMClient()
    executor = WorkflowExecutor(registry, store, llm)
    try:
        created = await executor.create_run(
            "sample", params={"goal": "ship"}, endpoint="node-a"
        )
        run_id = created["run"]["run_id"]
        await executor.advance(run_id)
        completed = await executor.advance(run_id)

        assert completed["run"]["status"] == "completed"
        assert [step["status"] for step in completed["steps"]] == [
            "completed",
            "completed",
        ]
        assert [artifact["step_id"] for artifact in completed["artifacts"]] == [
            "first",
            "second",
        ]

        retried = await executor.retry_step(run_id, "first")

        assert retried["run"]["status"] == "running"
        assert retried["run"]["completed_at"] is None
        assert [step["status"] for step in retried["steps"]] == ["pending", "pending"]
        assert retried["steps"][0]["output_json"] is None
        assert retried["steps"][1]["output_json"] is None
        assert retried["artifacts"] == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_executor_retry_completed_leaf_step_preserves_upstream_artifacts(tmp_path):
    write_workflow(tmp_path / "sample.yaml")
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    llm = FakeLLMClient()
    executor = WorkflowExecutor(registry, store, llm)
    try:
        created = await executor.create_run(
            "sample", params={"goal": "ship"}, endpoint="node-a"
        )
        run_id = created["run"]["run_id"]
        await executor.advance(run_id)
        await executor.advance(run_id)

        retried = await executor.retry_step(run_id, "second")

        assert [step["status"] for step in retried["steps"]] == [
            "completed",
            "pending",
        ]
        assert [artifact["step_id"] for artifact in retried["artifacts"]] == ["first"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_executor_does_not_complete_run_with_step_already_running(tmp_path):
    import asyncio

    write_workflow(tmp_path / "sample.yaml")
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    llm = BlockingLLMClient()
    executor = WorkflowExecutor(registry, store, llm)
    try:
        created = await executor.create_run(
            "sample", params={"goal": "ship"}, endpoint="node-a"
        )
        run_id = created["run"]["run_id"]

        in_flight = asyncio.create_task(executor.advance(run_id))
        while not llm.started:
            await asyncio.sleep(0)

        concurrent = await executor.advance(run_id)

        assert concurrent["run"]["status"] == "running"
        assert concurrent["steps"][0]["status"] == "running"
        assert concurrent["steps"][1]["status"] == "pending"

        llm.release.set()
        await in_flight
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
        created = await executor.create_run(
            "sample", params={"goal": "ship"}, endpoint="node-a"
        )
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


def test_render_template_json_escapes_string_values():
    rendered = render_template(
        "{{ json params.query }}",
        {"params": {"query": 'best "portable induction" cooktop'}},
    )

    assert rendered == '"best \\"portable induction\\" cooktop"'
