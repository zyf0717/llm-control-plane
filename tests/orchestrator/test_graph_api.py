import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.orchestrator.graph.api import create_graph_router
from src.orchestrator.graph.executor import GraphExecutor
from src.orchestrator.graph.registry import GraphRegistry
from src.orchestrator.graph.store import SQLiteGraphRunStore
from src.orchestrator.proxy import create_app


def write_fake_graph(path: Path) -> None:
    path.write_text(
        """
class FakeGraph:
    async def ainvoke(self, input, config=None):
        return {"answer": input["question"], "thread_id": config["configurable"]["thread_id"]}

    async def astream(self, input, config=None, stream_mode=None):
        yield ("updates", {"answer": {"question": input["question"]}})

graph = FakeGraph()
""",
        encoding="utf-8",
    )


def write_real_langgraph(path: Path) -> None:
    path.write_text(
        """
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict, total=False):
    question: str
    answer: str


def answer(state: State) -> State:
    return {"answer": state["question"].upper()}


builder = StateGraph(State)
builder.add_node("answer", answer)
builder.add_edge(START, "answer")
builder.add_edge("answer", END)
graph = builder.compile()
""",
        encoding="utf-8",
    )


def build_registry(tmp_path: Path) -> GraphRegistry:
    graph_module = tmp_path / "fake_graph.py"
    write_fake_graph(graph_module)
    metadata_dir = tmp_path / "graph_configs"
    metadata_dir.mkdir()
    (metadata_dir / "sample.yaml").write_text(
        """
id: sample
name: Sample
input_schema:
  type: object
  required: [question]
  properties:
    question:
      type: string
defaults:
  configurable:
    endpoint: primary
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
    return registry


@pytest.mark.asyncio
async def test_graph_executor_persists_run_and_streams_events(tmp_path):
    registry = build_registry(tmp_path)
    store = SQLiteGraphRunStore(tmp_path / "graphs.sqlite3")
    await store.initialize()
    try:
        executor = GraphExecutor(registry, store)
        created = await executor.create_run(
            "sample",
            input={"question": "ship"},
            config={"configurable": {"thread_id": "thread-a"}},
        )
        run_id = created["run_id"]

        events = [
            event async for event in executor.stream(run_id, stream_mode="updates")
        ]

        snapshot = await store.snapshot(run_id)
        assert events[0]["type"] == "graph_run_started"
        assert any(event.get("type") == "graph_event" for event in events)
        assert events[-1]["type"] == "graph_run_completed"
        assert snapshot["run"]["status"] == "completed"
        assert snapshot["run"]["thread_id"] == "thread-a"
        assert snapshot["events"][0]["node_name"] == "answer"
    finally:
        await store.close()


def test_graph_api_create_run_and_run(tmp_path):
    registry = build_registry(tmp_path)
    store = SQLiteGraphRunStore(tmp_path / "graphs.sqlite3")
    executor = GraphExecutor(registry, store)
    app = FastAPI()
    app.include_router(
        create_graph_router(
            registry_getter=lambda: registry,
            store_getter=lambda: store,
            executor_getter=lambda: executor,
        )
    )

    with TestClient(app) as client:
        client.portal.call(store.initialize)
        create_response = client.post(
            "/graphs/sample/runs",
            json={"input": {"question": "ship"}},
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        run_response = client.post(f"/graph-runs/{run_id}/run")

        assert run_response.status_code == 200
        assert run_response.json()["run"]["status"] == "completed"
        assert run_response.json()["run"]["output"]["answer"] == "ship"
        client.portal.call(store.close)


def test_graph_api_rejects_invalid_input(tmp_path):
    registry = build_registry(tmp_path)
    store = SQLiteGraphRunStore(tmp_path / "graphs.sqlite3")
    executor = GraphExecutor(registry, store)
    app = FastAPI()
    app.include_router(
        create_graph_router(
            registry_getter=lambda: registry,
            store_getter=lambda: store,
            executor_getter=lambda: executor,
        )
    )

    with TestClient(app) as client:
        client.portal.call(store.initialize)
        response = client.post("/graphs/sample/runs", json={"input": {}})
        assert response.status_code == 400
        assert "required property" in response.json()["detail"]
        client.portal.call(store.close)


def test_app_can_start_without_orchestration_subsystems():
    app = create_app(orchestration_subsystems=[])

    with TestClient(app) as client:
        response = client.get("/graphs")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_graph_executor_invokes_compiled_langgraph(tmp_path):
    pytest.importorskip("langgraph")
    graph_module = tmp_path / "real_graph.py"
    write_real_langgraph(graph_module)
    config_path = tmp_path / "langgraph.json"
    config_path.write_text(
        json.dumps({"graphs": {"real": f"{graph_module}:graph"}}),
        encoding="utf-8",
    )
    registry = GraphRegistry(config_path=config_path, metadata_dir=tmp_path / "missing")
    registry.load()
    store = SQLiteGraphRunStore(tmp_path / "graphs.sqlite3")
    await store.initialize()
    try:
        executor = GraphExecutor(registry, store)
        created = await executor.create_run("real", input={"question": "ship"})
        snapshot = await executor.run(created["run_id"])

        assert snapshot["run"]["status"] == "completed"
        assert snapshot["run"]["output"]["answer"] == "SHIP"
    finally:
        await store.close()
