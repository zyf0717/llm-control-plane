import json
from pathlib import Path
import ast

import pytest

from src.orchestrator.graph.registry import GraphRegistry


def write_fake_graph(path: Path) -> None:
    path.write_text(
        """
class FakeGraph:
    async def ainvoke(self, input, config=None):
        return {"input": input, "thread_id": config["configurable"]["thread_id"]}

    async def astream(self, input, config=None, stream_mode=None):
        yield ("updates", {"fake_node": {"input": input, "mode": stream_mode}})

graph = FakeGraph()
""",
        encoding="utf-8",
    )


def test_graph_registry_loads_langgraph_config_and_metadata(tmp_path):
    graph_module = tmp_path / "fake_graph.py"
    write_fake_graph(graph_module)
    metadata_dir = tmp_path / "graph_configs"
    metadata_dir.mkdir()
    (metadata_dir / "sample.yaml").write_text(
        """
id: sample
name: Sample Graph
description: Test graph
input_schema:
  type: object
  required: [question]
  properties:
    question:
      type: string
defaults:
  configurable:
    endpoint: primary
ui:
  supports_streaming: true
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "langgraph.json"
    config_path.write_text(
        json.dumps({"graphs": {"sample": f"{graph_module}:graph"}}),
        encoding="utf-8",
    )

    registry = GraphRegistry(config_path=config_path, metadata_dir=metadata_dir)
    registry.load()

    spec = registry.get("sample")
    assert spec.id == "sample"
    assert spec.name == "Sample Graph"
    assert spec.defaults["configurable"]["endpoint"] == "primary"
    assert hasattr(spec.graph, "ainvoke")
    assert registry.list()[0].summary_dict()["graph_ref"] == f"{graph_module}:graph"


def test_graph_registry_missing_config_loads_empty_registry(tmp_path):
    registry = GraphRegistry(config_path=tmp_path / "missing.json")

    registry.load()

    assert registry.list() == []
    with pytest.raises(KeyError, match="unknown graph"):
        registry.get("missing")


def test_graph_registry_rejects_metadata_id_mismatch(tmp_path):
    graph_module = tmp_path / "fake_graph.py"
    write_fake_graph(graph_module)
    metadata_dir = tmp_path / "graph_configs"
    metadata_dir.mkdir()
    (metadata_dir / "sample.yaml").write_text("id: other\n", encoding="utf-8")
    config_path = tmp_path / "langgraph.json"
    config_path.write_text(
        json.dumps({"graphs": {"sample": f"{graph_module}:graph"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        GraphRegistry(config_path=config_path, metadata_dir=metadata_dir).load()


def test_shipped_graph_registry_loads_workflow_derived_graphs():
    registry = GraphRegistry()

    registry.load()

    graph_ids = {spec.id for spec in registry.list()}
    assert {
        "implementation_plan",
        "research_brief",
        "threaded_search",
    }.issubset(graph_ids)


def test_shipped_graphs_do_not_import_workflow_package():
    graph_dir = Path("src/graphs")
    forbidden_roots = {
        "src.orchestrator.workflow",
        "orchestrator.workflow",
    }

    for path in graph_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = {alias.name for alias in node.names}
                assert not imported & forbidden_roots, path
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not any(
                    module == root or module.startswith(f"{root}.")
                    for root in forbidden_roots
                ), path
