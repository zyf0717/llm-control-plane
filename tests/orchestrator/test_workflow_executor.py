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


class SequenceLLMClient:
    def __init__(self, texts: list[str]):
        self.texts = list(texts)
        self.prompts = []

    async def complete(self, **kwargs):
        self.prompts.append(kwargs)
        text = self.texts.pop(0) if self.texts else ""
        return {"text": text, "metadata": {"endpoint": kwargs["endpoint"]}}


class StreamingLLMClient:
    def __init__(self, chunks=None):
        self.chunks = chunks or [
            {"channel": "content", "content": "stream "},
            {"channel": "content", "content": "answer"},
        ]
        self.prompts = []
        self.complete_calls = 0

    async def complete(self, **kwargs):
        self.complete_calls += 1
        self.prompts.append(kwargs)
        return {"text": "complete answer", "metadata": {"endpoint": kwargs["endpoint"]}}

    async def stream_complete(self, **kwargs):
        self.prompts.append(kwargs)
        for chunk in self.chunks:
            yield chunk


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


class RerankingSearchClient(CapturingSearchClient):
    def __init__(self):
        super().__init__()
        self.rerank_calls = []

    async def search(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "results": [
                {
                    "title": kwargs["query"],
                    "url": f"https://example.com/{len(self.calls)}",
                    "rank": len(self.calls),
                }
            ]
        }

    async def rerank_results(self, **kwargs):
        self.rerank_calls.append(kwargs)
        return {
            "results": list(reversed(kwargs["results"])),
            "reranking": {"used": True, "model": "stub", "path": "llm"},
            "warnings": [],
            "search_evidence": "reranked",
        }


class EmptySearchClient(CapturingSearchClient):
    async def search(self, **kwargs):
        self.calls.append(kwargs)
        return {"query": kwargs["query"], "results": [], "warnings": ["empty"]}

    async def rerank_results(self, **kwargs):
        raise AssertionError("reranker should not run without candidates")


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


def write_source_workflow(path: Path) -> None:
    path.write_text(
        """
id: source_sample
name: Source Sample
version: 0.1.0
params_schema:
  type: object
  required: [goal]
defaults:
  reasoning_effort: high
  search_provider: wikipedia_opensearch
steps:
  - id: search_evidence
    kind: search
    prompt: "{{ params.goal }}"
    output_key: search_evidence
  - id: synthesize
    kind: llm
    depends_on: [search_evidence]
    prompt: "{{ outputs.search_evidence.text }}"
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
    output_contract:
      format: json
      required: true
      schema:
        type: object
        additionalProperties: false
        required: [query]
        properties:
          query:
            type: string
            minLength: 1
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
    output_contract:
      format: json
      required: true
      schema:
        type: object
        additionalProperties: false
        required: [queries]
        properties:
          queries:
            type: array
            minItems: 1
            items:
              type: string
              minLength: 1
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


def write_explicit_search_controls_workflow(path: Path) -> None:
    path.write_text(
        """
id: explicit_search_controls_sample
name: Explicit Search Controls Sample
version: 0.1.0
params_schema:
  type: object
  required: [goal]
steps:
  - id: search
    kind: search
    search_count: 20
    use_query_refiner: false
    prompt: "{{ params.goal }}"
    output_key: search
  - id: rerank
    kind: rerank
    rerank_top_k: 10
    depends_on: [search]
    rerank_source_text: "Goal: {{ params.goal }}"
    prompt: "{{ params.goal }}"
    output_key: search
""",
        encoding="utf-8",
    )


def write_structured_plan_workflow(
    path: Path,
    *,
    on_invalid: str = "retry",
    max_attempts: int = 2,
    repair: bool = True,
) -> None:
    path.write_text(
        f"""
id: structured_plan_sample
name: Structured Plan Sample
version: 0.1.0
params_schema:
  type: object
  required: [goal]
steps:
  - id: plan
    kind: llm
    prompt: "{{{{ params.goal }}}}"
    output_key: plan
    output_contract:
      format: json
      required: true
      schema:
        type: object
        additionalProperties: false
        required: [queries]
        properties:
          queries:
            type: array
            minItems: 1
            items:
              type: string
              minLength: 1
      on_invalid:
        action: {on_invalid}
        max_attempts: {max_attempts}
        repair: {str(repair).lower()}
""",
        encoding="utf-8",
    )


def write_json_queries_rerank_workflow(path: Path) -> None:
    path.write_text(
        """
id: json_queries_rerank_sample
name: JSON Queries Rerank Sample
version: 0.1.0
params_schema:
  type: object
  required: [goal]
steps:
  - id: plan
    kind: llm
    prompt: "{{ params.goal }}"
    output_key: plan
    output_contract:
      format: json
      required: true
      schema:
        type: object
        additionalProperties: false
        required: [queries]
        properties:
          queries:
            type: array
            minItems: 1
            items:
              type: string
              minLength: 1
  - id: search
    kind: search
    depends_on: [plan]
    search_count: 20
    use_query_refiner: false
    prompt: "{{ outputs.plan.json.queries }}"
    output_key: search
  - id: rerank
    kind: rerank
    rerank_top_k: 10
    depends_on: [search]
    prompt: "{{ params.goal }}"
    output_key: search
""",
        encoding="utf-8",
    )


def write_visible_llm_workflow(path: Path, *, visibility: str = "final") -> None:
    path.write_text(
        f"""
id: visible_sample
name: Visible Sample
version: 0.1.0
params_schema:
  type: object
  required: [goal]
steps:
  - id: first
    kind: llm
    chat_visibility: {visibility}
    prompt: "{{{{ params.goal }}}}"
    output_key: first
""",
        encoding="utf-8",
    )


def write_visible_search_workflow(path: Path) -> None:
    path.write_text(
        """
id: visible_search_sample
name: Visible Search Sample
version: 0.1.0
params_schema:
  type: object
  required: [goal]
steps:
  - id: search
    kind: search
    chat_visibility: intermediate
    prompt: "{{ params.goal }}"
    output_key: search
""",
        encoding="utf-8",
    )


def write_compress_source_workflow(
    path: Path,
    *,
    input_format: str = "auto",
    output_format: str = "text",
    trigger: int = 1000,
    chunk: int = 800,
    target: int = 400,
    max_output: int = 600,
    max_json_bytes: int = 12000,
    max_rounds: int = 2,
    output_contract: str = "",
) -> None:
    path.write_text(
        f"""
id: compress_sample
name: Compress Sample
version: 0.1.0
params_schema:
  type: object
  required: [manual_source_text]
steps:
  - id: compress
    kind: compress_source
    prompt: "{{{{ params.manual_source_text }}}}"
    output_key: compressed
    compression_input_format: {input_format}
    compression_output_format: {output_format}
    compression_trigger_chars: {trigger}
    compression_chunk_chars: {chunk}
    compression_target_chars: {target}
    compression_max_output_chars: {max_output}
    compression_max_output_json_bytes: {max_json_bytes}
    compression_max_rounds: {max_rounds}
    compression_goal: Preserve evidence, source refs, and concept logic.
{output_contract}
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
async def test_executor_uses_run_search_provider_and_retrieval_endpoint(tmp_path):
    write_source_workflow(tmp_path / "source_sample.yaml")
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    llm = FakeLLMClient()
    search = CapturingSearchClient()
    executor = WorkflowExecutor(registry, store, llm, search)
    try:
        created = await executor.create_run(
            "source_sample",
            params={"goal": "ship"},
            endpoint="node-a",
            retrieval_endpoint="http://retrieval/api/retrieve/context",
            search_provider="duckduckgo_html",
        )
        run_id = created["run"]["run_id"]

        await executor.advance(run_id)
        await executor.advance(run_id)

        assert search.calls[0]["provider"] == "duckduckgo_html"
        assert llm.prompts[0]["retrieval_endpoint"] == "http://retrieval/api/retrieve/context"
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
async def test_search_step_fans_out_workflow_planned_queries_without_query_refiner(tmp_path):
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
        assert [call["use_query_refiner"] for call in search.calls] == [False, False]
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
async def test_llm_structured_output_contract_passes_and_persists_metadata(tmp_path):
    write_structured_plan_workflow(tmp_path / "structured.yaml")
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    llm = JsonLLMClient('{"queries": ["query one"]}')
    executor = WorkflowExecutor(registry, store, llm)
    try:
        created = await executor.create_run(
            "structured_plan_sample", params={"goal": "ship"}, endpoint="node-a"
        )
        snapshot = await executor.advance(created["run"]["run_id"])

        output = snapshot["steps"][0]["output_json"]
        assert output["json"] == {"queries": ["query one"]}
        assert output["metadata"]["structured_output"] == {
            "format": "json",
            "valid": True,
            "schema_enforced": True,
            "attempts": 1,
            "repair_used": False,
            "errors": [],
        }
        assert "You must return only valid JSON" in llm.prompts[0]["prompt"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_llm_structured_output_contract_fails_without_repair_budget(tmp_path):
    write_structured_plan_workflow(
        tmp_path / "structured.yaml",
        on_invalid="fail",
        max_attempts=0,
        repair=False,
    )
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    llm = JsonLLMClient('{"queries": "query one"}')
    executor = WorkflowExecutor(registry, store, llm)
    try:
        created = await executor.create_run(
            "structured_plan_sample", params={"goal": "ship"}, endpoint="node-a"
        )
        snapshot = await executor.advance(created["run"]["run_id"])

        assert snapshot["run"]["status"] == "failed"
        assert snapshot["steps"][0]["status"] == "failed"
        assert "structured_output_invalid" in snapshot["steps"][0]["error"]
        assert len(llm.prompts) == 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_llm_structured_output_repairs_then_retries_within_budget(tmp_path):
    write_structured_plan_workflow(tmp_path / "structured.yaml")
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    llm = SequenceLLMClient(
        [
            '{"queries": "query one"}',
            '{"queries": "still wrong"}',
            '{"queries": ["query one"]}',
        ]
    )
    executor = WorkflowExecutor(registry, store, llm)
    try:
        created = await executor.create_run(
            "structured_plan_sample", params={"goal": "ship"}, endpoint="node-a"
        )
        snapshot = await executor.advance(created["run"]["run_id"])

        output = snapshot["steps"][0]["output_json"]
        assert snapshot["run"]["status"] == "completed"
        assert output["json"] == {"queries": ["query one"]}
        assert output["metadata"]["structured_output"]["attempts"] == 3
        assert output["metadata"]["structured_output"]["repair_used"] is True
        assert "Previous output:" in llm.prompts[1]["prompt"]
        assert "previous response was invalid" in llm.prompts[2]["prompt"]
        assert all(prompt["skip_conversation"] is True for prompt in llm.prompts)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_llm_structured_output_defaults_to_repair_then_retry(tmp_path):
    path = tmp_path / "structured_default.yaml"
    path.write_text(
        """
id: structured_default_sample
name: Structured Default Sample
version: 0.1.0
params_schema:
  type: object
  required: [goal]
steps:
  - id: plan
    kind: llm
    prompt: "{{ params.goal }}"
    output_key: plan
    output_contract:
      format: json
      required: true
      schema:
        type: object
        required: [queries]
        properties:
          queries:
            type: array
""",
        encoding="utf-8",
    )
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    llm = SequenceLLMClient(
        [
            '{"queries": "query one"}',
            '{"queries": "still wrong"}',
            '{"queries": ["query one"]}',
        ]
    )
    executor = WorkflowExecutor(registry, store, llm)
    try:
        created = await executor.create_run(
            "structured_default_sample", params={"goal": "ship"}, endpoint="node-a"
        )
        snapshot = await executor.advance(created["run"]["run_id"])

        assert snapshot["run"]["status"] == "completed"
        assert "Previous output:" in llm.prompts[1]["prompt"]
        assert "previous response was invalid" in llm.prompts[2]["prompt"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_search_step_can_dispatch_json_string_query_without_query_refiner(tmp_path):
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
                "count": 5,
                "use_query_refiner": False,
            }
        ]
        assert llm.prompts == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_rerank_step_honors_explicit_source_text_and_overwrites_output_key(tmp_path):
    write_explicit_search_controls_workflow(
        tmp_path / "explicit_search_controls_sample.yaml"
    )
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    search = RerankingSearchClient()
    executor = WorkflowExecutor(registry, store, FakeLLMClient(), search)
    try:
        created = await executor.create_run(
            "explicit_search_controls_sample",
            params={"goal": "ship"},
            endpoint="node-a",
        )
        run_id = created["run"]["run_id"]

        await executor.advance(run_id)
        snapshot = await executor.advance(run_id)

        assert search.calls == [
            {
                "query": "ship",
                "provider": None,
                "count": 20,
                "use_query_refiner": False,
            }
        ]
        assert search.rerank_calls == [
            {
                "query": "ship",
                "results": [
                    {
                        "title": "ship",
                        "url": "https://example.com/1",
                        "rank": 1,
                    }
                ],
                "source_text": "Goal: ship",
                "top_k": 10,
            }
        ]
        outputs = WorkflowExecutor._previous_outputs(
            registry.get("explicit_search_controls_sample"),
            snapshot,
        )
        assert outputs["search"]["metadata"]["kind"] == "rerank"
        assert outputs["search"]["json"]["workflow_rerank"]["source_output"] == "search"
        assert outputs["search"]["json"]["workflow_rerank"]["path"] == "llm"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_rerank_step_passes_through_empty_search_results(tmp_path):
    write_explicit_search_controls_workflow(
        tmp_path / "explicit_search_controls_sample.yaml"
    )
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    search = EmptySearchClient()
    executor = WorkflowExecutor(registry, store, FakeLLMClient(), search)
    try:
        created = await executor.create_run(
            "explicit_search_controls_sample",
            params={"goal": "ship"},
            endpoint="node-a",
        )
        run_id = created["run"]["run_id"]

        await executor.advance(run_id)
        snapshot = await executor.advance(run_id)

        output = snapshot["steps"][1]["output_json"]["json"]
        assert output["results"] == []
        assert output["warnings"] == ["empty"]
        assert output["reranking"]["path"] == "none"
        assert output["workflow_rerank"]["skipped"] == "no-results"
        assert output["workflow_rerank"]["path"] == "none"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_multi_query_workflow_reranks_merged_results_when_available(tmp_path):
    write_json_queries_rerank_workflow(tmp_path / "json_queries_rerank_sample.yaml")
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    llm = JsonLLMClient('{"queries": ["query one", "query two"]}')
    search = RerankingSearchClient()
    executor = WorkflowExecutor(registry, store, llm, search)
    try:
        created = await executor.create_run(
            "json_queries_rerank_sample", params={"goal": "ship"}, endpoint="node-a"
        )
        run_id = created["run"]["run_id"]

        await executor.advance(run_id)
        await executor.advance(run_id)
        snapshot = await executor.advance(run_id)

        assert [call["count"] for call in search.calls] == [20, 20]
        assert search.rerank_calls[0]["query"] == "ship"
        assert search.rerank_calls[0]["top_k"] == 10
        search_output = snapshot["steps"][2]["output_json"]["json"]
        assert search_output["reranking"] == {
            "used": True,
            "model": "stub",
            "path": "llm",
        }
        assert search_output["workflow_rerank"]["source_output"] == "search"
        assert search_output["workflow_rerank"]["path"] == "llm"
        assert [item["title"] for item in search_output["results"]] == [
            "query two",
            "query one",
        ]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_compress_source_passes_through_below_trigger_without_llm(tmp_path):
    write_compress_source_workflow(tmp_path / "compress.yaml", trigger=1000)
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    llm = FakeLLMClient()
    executor = WorkflowExecutor(registry, store, llm)
    try:
        created = await executor.create_run(
            "compress_sample",
            params={"manual_source_text": "short source text with useful evidence"},
            endpoint="node-a",
        )
        snapshot = await executor.advance(created["run"]["run_id"])

        output = snapshot["steps"][0]["output_json"]
        assert llm.prompts == []
        assert output["text"] == "short source text with useful evidence"
        assert output["json"]["compressed"] is False
        assert output["json"]["method"] == "pass_through"
        assert output["metadata"]["kind"] == "compress_source"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_compress_source_pass_through_enforces_output_payload_cap(tmp_path):
    write_compress_source_workflow(
        tmp_path / "compress.yaml",
        trigger=1000,
        max_json_bytes=120,
    )
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    llm = FakeLLMClient()
    executor = WorkflowExecutor(registry, store, llm)
    try:
        created = await executor.create_run(
            "compress_sample",
            params={"manual_source_text": "short"},
            endpoint="node-a",
        )
        snapshot = await executor.advance(created["run"]["run_id"])

        assert llm.prompts == []
        assert snapshot["run"]["status"] == "failed"
        assert "compression_output_over_budget" in snapshot["steps"][0]["error"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_compress_source_rejects_malformed_explicit_json_input(tmp_path):
    write_compress_source_workflow(
        tmp_path / "compress.yaml",
        input_format="json",
    )
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    llm = FakeLLMClient()
    executor = WorkflowExecutor(registry, store, llm)
    try:
        created = await executor.create_run(
            "compress_sample",
            params={"manual_source_text": '{"bad": '},
            endpoint="node-a",
        )
        snapshot = await executor.advance(created["run"]["run_id"])

        assert llm.prompts == []
        assert snapshot["run"]["status"] == "failed"
        assert "compression_input_shape_invalid" in snapshot["steps"][0]["error"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_compress_source_calls_llm_above_trigger_and_preserves_quality_prompt(
    tmp_path,
):
    write_compress_source_workflow(
        tmp_path / "compress.yaml",
        trigger=500,
        chunk=400,
        target=300,
        max_output=450,
    )
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    compress = (
        '"ACME Cooling Vest" 2025 42% https://example.gov/study example.gov '
        "ACME Independent Review OSHA outdoor workers evidence caveat."
    )
    llm = SequenceLLMClient([compress] * 12)
    executor = WorkflowExecutor(registry, store, llm)
    source = (
        '"ACME Cooling Vest" showed 42% reduction in 2025 '
        "https://example.gov/study with independent evidence.\n\n"
        "Independent Review connected the finding to OSHA outdoor workers guidance "
        "and noted caveats. "
    ) * 4
    try:
        created = await executor.create_run(
            "compress_sample", params={"manual_source_text": source}, endpoint="node-a"
        )
        snapshot = await executor.advance(created["run"]["run_id"])

        first_prompt = llm.prompts[0]["prompt"]
        assert "claim -> evidence -> caveat -> source_ref" in first_prompt
        assert "untrusted evidence" in first_prompt
        assert "Highest-value evidence candidates" in first_prompt
        output = snapshot["steps"][0]["output_json"]
        assert snapshot["run"]["status"] == "completed"
        assert len(llm.prompts) >= output["json"]["chunks"]
        assert output["json"]["compressed"] is True
        assert output["json"]["warnings"] == []
        assert "https://example.gov/study" in output["text"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_compress_source_repairs_malformed_structured_output(tmp_path):
    write_compress_source_workflow(
        tmp_path / "compress.yaml",
        output_format="json",
        trigger=240,
        chunk=240,
        target=180,
        max_output=220,
    )
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    valid = (
        '{"summary":"x","preserved_keywords":[],"evidence_snippets":[],'
        '"uncertainties":[],"source_refs":[]}'
    )
    llm = SequenceLLMClient(["not-json", valid, *([valid] * 8)])
    executor = WorkflowExecutor(registry, store, llm)
    try:
        created = await executor.create_run(
            "compress_sample",
            params={"manual_source_text": ("alpha beta evidence. " * 20)},
            endpoint="node-a",
        )
        snapshot = await executor.advance(created["run"]["run_id"])

        output = snapshot["steps"][0]["output_json"]
        assert snapshot["run"]["status"] == "completed"
        assert "output_repaired" in output["json"]["warnings"]
        assert output["json"]["compressed_output"]["summary"] == "x"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_compress_source_fails_when_repaired_output_still_over_budget(tmp_path):
    write_compress_source_workflow(
        tmp_path / "compress.yaml",
        trigger=240,
        chunk=240,
        target=100,
        max_output=120,
    )
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    llm = SequenceLLMClient(["x" * 200, "y" * 200])
    executor = WorkflowExecutor(registry, store, llm)
    try:
        created = await executor.create_run(
            "compress_sample",
            params={"manual_source_text": ("alpha beta evidence. " * 20)},
            endpoint="node-a",
        )
        snapshot = await executor.advance(created["run"]["run_id"])

        assert snapshot["run"]["status"] == "failed"
        assert "compression_output_over_budget" in snapshot["steps"][0]["error"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_compress_source_reduces_multi_chunk_summaries(tmp_path):
    write_compress_source_workflow(
        tmp_path / "compress.yaml",
        trigger=240,
        chunk=240,
        target=90,
        max_output=160,
        max_rounds=2,
    )
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    llm = SequenceLLMClient(
        [
            "alpha evidence summary " * 4,
            "beta evidence summary " * 4,
            "gamma evidence summary " * 4,
            "delta evidence summary " * 4,
            *("alpha beta gamma compress evidence." for _ in range(8)),
        ]
    )
    executor = WorkflowExecutor(registry, store, llm)
    try:
        created = await executor.create_run(
            "compress_sample",
            params={"manual_source_text": ("alpha beta gamma evidence. " * 30)},
            endpoint="node-a",
        )
        snapshot = await executor.advance(created["run"]["run_id"])

        output = snapshot["steps"][0]["output_json"]
        assert snapshot["run"]["status"] == "completed"
        assert "compress evidence" in output["text"]
        assert output["json"]["rounds"] >= 1
        assert len(llm.prompts) > output["json"]["chunks"]
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
async def test_streaming_llm_step_accumulates_output_and_emits_deltas(tmp_path):
    write_visible_llm_workflow(tmp_path / "visible.yaml", visibility="final")
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    llm = StreamingLLMClient()
    executor = WorkflowExecutor(registry, store, llm)
    try:
        created = await executor.create_run(
            "visible_sample", params={"goal": "ship"}, endpoint="node-a"
        )
        run_id = created["run"]["run_id"]

        events = [
            event
            async for event in executor.run_to_completion_stream(
                run_id, stream_llm=True
            )
        ]

        deltas = [event for event in events if event["type"] == "step_delta"]
        assert [event["content"] for event in deltas] == ["stream ", "answer"]
        assert all(event["chat_visibility"] == "final" for event in deltas)
        snapshot = events[-1]["snapshot"]
        assert snapshot["run"]["status"] == "completed"
        assert snapshot["steps"][0]["output_json"]["text"] == "stream answer"
        assert llm.prompts[0]["skip_conversation"] is True
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_streaming_hidden_step_does_not_emit_chat_deltas(tmp_path):
    write_visible_llm_workflow(tmp_path / "hidden.yaml", visibility="hidden")
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    llm = StreamingLLMClient()
    executor = WorkflowExecutor(registry, store, llm)
    try:
        created = await executor.create_run(
            "visible_sample", params={"goal": "ship"}, endpoint="node-a"
        )
        run_id = created["run"]["run_id"]

        events = [
            event
            async for event in executor.run_to_completion_stream(
                run_id, stream_llm=True
            )
        ]

        assert not [event for event in events if event["type"] == "step_delta"]
        completed = [event for event in events if event["type"] == "step_completed"]
        assert completed[0]["content"] == ""
        assert llm.complete_calls == 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_non_stream_visible_llm_emits_completion_content_once(tmp_path):
    write_visible_llm_workflow(tmp_path / "visible.yaml", visibility="intermediate")
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    llm = StreamingLLMClient()
    executor = WorkflowExecutor(registry, store, llm)
    try:
        created = await executor.create_run(
            "visible_sample", params={"goal": "ship"}, endpoint="node-a"
        )
        run_id = created["run"]["run_id"]

        events = [
            event
            async for event in executor.run_to_completion_stream(
                run_id, stream_llm=False
            )
        ]

        assert not [event for event in events if event["type"] == "step_delta"]
        completed = [event for event in events if event["type"] == "step_completed"]
        assert completed[0]["chat_visibility"] == "intermediate"
        assert completed[0]["content"] == "complete answer"
        assert completed[0]["streamed"] is False
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_visible_search_step_emits_completion_content_only(tmp_path):
    write_visible_search_workflow(tmp_path / "visible_search.yaml")
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    search = CapturingSearchClient()
    executor = WorkflowExecutor(registry, store, FakeLLMClient(), search)
    try:
        created = await executor.create_run(
            "visible_search_sample", params={"goal": "ship"}, endpoint="node-a"
        )
        run_id = created["run"]["run_id"]

        events = [
            event
            async for event in executor.run_to_completion_stream(
                run_id, stream_llm=True
            )
        ]

        assert not [event for event in events if event["type"] == "step_delta"]
        completed = [event for event in events if event["type"] == "step_completed"]
        assert completed[0]["chat_visibility"] == "intermediate"
        assert '"results"' in completed[0]["content"]
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
