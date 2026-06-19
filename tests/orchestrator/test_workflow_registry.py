import re
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


def test_registry_loads_output_contract(tmp_path):
    write_workflow(
        tmp_path / "sample.yaml",
        body=minimal_workflow(
            steps="""
  - id: first
    kind: llm
    prompt: hello
    output_contract:
      format: json
      required: true
      schema:
        type: object
      on_invalid:
        action: retry
        max_attempts: 2
        repair: true
"""
        ),
    )

    registry = WorkflowRegistry(tmp_path)
    registry.load()

    contract = registry.get("sample").steps[0].output_contract
    assert contract is not None
    assert contract.format == "json"
    assert contract.required is True
    assert contract.schema == {"type": "object"}
    assert contract.on_invalid == {
        "action": "retry",
        "max_attempts": 2,
        "repair": True,
    }


def test_registry_rejects_output_schema(tmp_path):
    write_workflow(
        tmp_path / "sample.yaml",
        body=minimal_workflow(
            steps="""
  - id: first
    kind: llm
    prompt: hello
    output_schema:
      type: object
"""
        ),
    )

    with pytest.raises(ValueError, match="output_schema is no longer supported"):
        WorkflowRegistry(tmp_path).load()


def test_registry_rejects_invalid_output_contract_format(tmp_path):
    write_workflow(
        tmp_path / "sample.yaml",
        body=minimal_workflow(
            steps="""
  - id: first
    kind: llm
    prompt: hello
    output_contract:
      format: xml
"""
        ),
    )

    with pytest.raises(ValueError, match="output_contract.format"):
        WorkflowRegistry(tmp_path).load()


def test_builtin_json_consumers_have_producer_contracts():
    registry = WorkflowRegistry(DEFAULT_WORKFLOW_DIR)
    registry.load()

    for spec in registry.list():
        producers = {
            step.output_key or step.id: step
            for step in spec.steps
        }
        for step in spec.steps:
            for output_key in re.findall(
                r"outputs\.([A-Za-z_][A-Za-z0-9_]*)\.json",
                step.prompt or "",
            ):
                producer = producers.get(output_key)
                assert producer is not None, (
                    f"{spec.id}.{step.id} consumes unknown JSON output {output_key}"
                )
                assert producer.output_contract is not None, (
                    f"{spec.id}.{step.id} consumes outputs.{output_key}.json, "
                    f"but producer {producer.id} has no output_contract"
                )


def test_registry_loads_rerank_steps(tmp_path):
    write_workflow(
        tmp_path / "sample.yaml",
        body=minimal_workflow(
            steps="""
  - id: search
    name: Search
    kind: search
    use_query_refiner: false
    prompt: hello
  - id: rerank
    name: Rerank
    kind: rerank
    depends_on: [search]
    rerank_source_text: "Goal: {{ params.goal }}"
    prompt: hello
"""
        ),
    )

    registry = WorkflowRegistry(tmp_path)
    registry.load()

    search_step = registry.get("sample").steps[0]
    rerank_step = registry.get("sample").steps[1]
    assert search_step.use_query_refiner is False
    assert rerank_step.kind == "rerank"
    assert rerank_step.depends_on == ["search"]
    assert rerank_step.rerank_source_text == "Goal: {{ params.goal }}"


def test_registry_loads_compress_source_steps(tmp_path):
    write_workflow(
        tmp_path / "sample.yaml",
        body=minimal_workflow(
            steps="""
  - id: compress
    name: Compress
    kind: compress_source
    prompt: "{{ params.manual_source_text }}"
    output_key: compressed
    compression_trigger_chars: 1000
    compression_chunk_chars: 800
    compression_target_chars: 400
    compression_max_output_chars: 600
    compression_max_output_json_bytes: 12000
    compression_max_rounds: 2
    compression_input_format: auto
    compression_output_format: text
    compression_goal: Preserve evidence.
"""
        ),
    )

    registry = WorkflowRegistry(tmp_path)
    registry.load()

    step = registry.get("sample").steps[0]
    assert step.kind == "compress_source"
    assert step.compression_trigger_chars == 1000
    assert step.compression_chunk_chars == 800
    assert step.compression_target_chars == 400
    assert step.compression_max_output_chars == 600
    assert step.compression_max_output_json_bytes == 12000
    assert step.compression_max_rounds == 2
    assert step.compression_input_format == "auto"
    assert step.compression_output_format == "text"
    assert step.compression_goal == "Preserve evidence."


def test_registry_loads_repo_context_steps(tmp_path):
    write_workflow(
        tmp_path / "sample.yaml",
        body=minimal_workflow(
            steps="""
  - id: explore
    name: Explore
    kind: repo_context
    prompt: "{{ params.query }}"
    repo_context_repo: "{{ params.repo_name }}"
    repo_context_max_turns: 4
    output_key: repo_context
"""
        ),
    )

    registry = WorkflowRegistry(tmp_path)
    registry.load()

    step = registry.get("sample").steps[0]
    assert step.kind == "repo_context"
    assert step.repo_context_repo == "{{ params.repo_name }}"
    assert step.repo_context_max_turns == 4


def test_registry_rejects_repo_context_fields_on_other_steps(tmp_path):
    write_workflow(
        tmp_path / "sample.yaml",
        body=minimal_workflow(
            steps="""
  - id: first
    kind: llm
    prompt: hello
    repo_context_repo: sample
"""
        ),
    )

    with pytest.raises(ValueError, match="repo_context fields"):
        WorkflowRegistry(tmp_path).load()


def test_registry_rejects_invalid_compression_budgets(tmp_path):
    write_workflow(
        tmp_path / "sample.yaml",
        body=minimal_workflow(
            steps="""
  - id: compress
    kind: compress_source
    prompt: hello
    compression_trigger_chars: 100
    compression_chunk_chars: 80
    compression_target_chars: 90
    compression_max_output_chars: 120
"""
        ),
    )

    with pytest.raises(ValueError, match="compression budgets"):
        WorkflowRegistry(tmp_path).load()


def test_registry_rejects_zero_compression_budgets(tmp_path):
    write_workflow(
        tmp_path / "sample.yaml",
        body=minimal_workflow(
            steps="""
  - id: compress
    kind: compress_source
    prompt: hello
    compression_max_rounds: 0
"""
        ),
    )

    with pytest.raises(ValueError, match="compression_max_rounds must be positive"):
        WorkflowRegistry(tmp_path).load()


def test_registry_rejects_invalid_compression_formats(tmp_path):
    write_workflow(
        tmp_path / "sample.yaml",
        body=minimal_workflow(
            steps="""
  - id: compress
    kind: compress_source
    prompt: hello
    compression_input_format: xml
"""
        ),
    )

    with pytest.raises(ValueError, match="compression_input_format"):
        WorkflowRegistry(tmp_path).load()


def test_registry_supplies_builtin_compression_contract_for_structured_output(tmp_path):
    write_workflow(
        tmp_path / "sample.yaml",
        body=minimal_workflow(
            steps="""
  - id: compress
    kind: compress_source
    prompt: hello
    compression_output_format: json
"""
        ),
    )

    registry = WorkflowRegistry(tmp_path)
    registry.load()

    contract = registry.get("sample").steps[0].output_contract
    assert contract is not None
    assert contract.format == "json"
    assert contract.required is True
    assert contract.schema is not None
    assert contract.schema["required"] == [
        "summary",
        "preserved_keywords",
        "evidence_snippets",
        "uncertainties",
        "source_refs",
    ]


def test_registry_rejects_compression_contract_format_mismatch(tmp_path):
    write_workflow(
        tmp_path / "sample.yaml",
        body=minimal_workflow(
            steps="""
  - id: compress
    kind: compress_source
    prompt: hello
    compression_output_format: json
    output_contract:
      format: yaml
"""
        ),
    )

    with pytest.raises(ValueError, match="output_contract.format"):
        WorkflowRegistry(tmp_path).load()


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


def test_registry_rejects_inline_workflow_reranker_flag(tmp_path):
    write_workflow(
        tmp_path / "sample.yaml",
        body=minimal_workflow(
            steps="""
  - id: search
    kind: search
    use_reranker: true
    prompt: hello
"""
        ),
    )

    with pytest.raises(ValueError, match="use_reranker"):
        WorkflowRegistry(tmp_path).load()


def test_registry_rejects_rerank_source_text_on_search_step(tmp_path):
    write_workflow(
        tmp_path / "sample.yaml",
        body=minimal_workflow(
            steps="""
  - id: search
    kind: search
    rerank_source_text: hello
    prompt: hello
"""
        ),
    )

    with pytest.raises(ValueError, match="rerank_source_text"):
        WorkflowRegistry(tmp_path).load()


def test_registry_rejects_old_compact_context_step_kind(tmp_path):
    write_workflow(
        tmp_path / "sample.yaml",
        body=minimal_workflow(
            steps="""
  - id: old_compress
    kind: compact_context
    prompt: hello
"""
        ),
    )

    with pytest.raises(ValueError, match="unsupported kind: compact_context"):
        WorkflowRegistry(tmp_path).load()


def test_registry_rejects_old_compaction_fields(tmp_path):
    write_workflow(
        tmp_path / "sample.yaml",
        body=minimal_workflow(
            steps="""
  - id: compress
    kind: compress_source
    prompt: hello
    compaction_target_chars: 100
"""
        ),
    )

    with pytest.raises(ValueError, match="compaction fields"):
        WorkflowRegistry(tmp_path).load()


def test_default_workflow_config_directory_loads_shipped_specs():
    assert DEFAULT_WORKFLOW_DIR.name == "workflow_configs"

    registry = WorkflowRegistry()
    registry.load()

    assert {spec.id for spec in registry.list()} == {
        "threaded_search",
        "implementation_plan",
        "research_brief",
        "repo_context",
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
