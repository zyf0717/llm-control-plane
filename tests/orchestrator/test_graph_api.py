import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.orchestrator.graph.api import create_graph_router
from src.orchestrator.graph.executor import GraphExecutor
from src.orchestrator.graph.registry import GraphRegistry
from src.orchestrator.graph.store import SQLiteGraphRunStore
from src.orchestrator.proxy import create_app
from src.graphs import common as graph_common
from src.llm_metrics import CallMetrics


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
@pytest.mark.parametrize("mode", ["run", "stream", "resume", "failure"])
async def test_graph_records_all_calls_and_streams_completed_call_metrics(tmp_path, mode):
    class MeasuredGraph:
        async def ainvoke(self, input, config=None):
            with CallMetrics(label="Planner"):
                pass
            with CallMetrics(label="Answer"):
                if mode == "failure":
                    raise RuntimeError("lost")
            return {"answer": "done"}

        async def astream(self, input, config=None, stream_mode=None):
            with CallMetrics(label="Planner"):
                pass
            yield {"plan": {"text": "plan"}}
            with CallMetrics(label="Answer"):
                pass
            yield {"answer": {"text": "done"}}

    registry = build_registry(tmp_path)
    registry.get("sample").graph = MeasuredGraph()
    store = SQLiteGraphRunStore(tmp_path / "graphs.sqlite3")
    await store.initialize()
    try:
        executor = GraphExecutor(registry, store)
        run_id = (await executor.create_run("sample", input={"question": "hi"}))["run_id"]
        if mode == "stream":
            events = [event async for event in executor.stream(run_id)]
            intermediate = [event["snapshot"] for event in events if event.get("type") == "snapshot"]
            assert any(len([row for row in snap["events"] if row["event_type"] == "llm_call"]) == 1
                       for snap in intermediate)
            snapshot = events[-1]["snapshot"]
        elif mode == "resume":
            snapshot = await executor.resume(run_id, resume="yes")
        else:
            snapshot = await executor.run(run_id)
        calls = [row["event"]["call"] for row in snapshot["events"] if row["event_type"] == "llm_call"]
        assert [call["label"] for call in calls] == ["Planner", "Answer"]
        assert len({call["id"] for call in calls}) == 2
        assert calls[-1]["status"] == ("Failed" if mode == "failure" else "Completed")
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_graph_does_not_skip_parallel_calls_finishing_during_persistence(tmp_path):
    gate = asyncio.Event()

    class ParallelGraph:
        async def astream(self, *args, **kwargs):
            async def delayed_call():
                await gate.wait()
                with CallMetrics(label="parallel"):
                    pass
            self.pending = asyncio.create_task(delayed_call())
            with CallMetrics(label="first"):
                pass
            yield {"first": {}}
            await self.pending
            yield {"second": {}}

    graph = ParallelGraph()
    registry = build_registry(tmp_path)
    registry.get("sample").graph = graph
    store = SQLiteGraphRunStore(tmp_path / "graphs.sqlite3")
    await store.initialize()
    try:
        executor = GraphExecutor(registry, store)
        record = executor._record_calls

        async def slow_record(run_id, calls):
            gate.set()
            await graph.pending
            await record(run_id, calls)

        executor._record_calls = slow_record
        run_id = (await executor.create_run("sample", input={"question": "hi"}))["run_id"]
        events = [event async for event in executor.stream(run_id)]
        calls = [row["event"]["call"] for row in events[-1]["snapshot"]["events"]
                 if row["event_type"] == "llm_call"]
        assert [call["label"] for call in calls] == ["first", "parallel"]
    finally:
        await store.close()


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


class FakeGraphLLMClient:
    async def complete(self, **kwargs):
        prompt = kwargs["prompt"]
        if "Return strict JSON only" in prompt and '"retrieval_queries"' in prompt:
            return {
                "text": json.dumps(
                    {
                        "retrieval_queries": ["ship graph support"],
                        "retrieval_prompt": "Answer ship graph support.",
                        "reason": "test",
                    }
                ),
                "metadata": {"endpoint": kwargs["endpoint"], "workload_type": kwargs.get("workload_type")},
            }
        if "Produce one repo-context query for citation retrieval." in prompt:
            return {
                "text": json.dumps({"query": "Find graph support code."}),
                "metadata": {"endpoint": kwargs["endpoint"], "workload_type": kwargs.get("workload_type")},
            }
        if "Return strict JSON only" in prompt or "corrected strict JSON" in prompt:
            return {
                "text": json.dumps(
                    {
                        "rerank_query": "ship graph support",
                        "queries": ["ship graph support"],
                        "reason": "test",
                        "source_preferences": ["primary sources"],
                    }
                ),
                "metadata": {"endpoint": kwargs["endpoint"], "workload_type": kwargs.get("workload_type")},
            }
        return {
            "text": f"answer: {prompt[:20]}",
            "metadata": {"endpoint": kwargs["endpoint"], "workload_type": kwargs.get("workload_type")},
        }


class FakeGraphSearchClient:
    async def search(self, **kwargs):
        return {
            "query": kwargs["query"],
            "provider": kwargs.get("provider") or "fake",
            "results": [
                {
                    "title": "Result",
                    "url": "https://example.com",
                    "snippet": "Evidence",
                }
            ],
            "warnings": [],
        }

    async def rerank_results(self, **kwargs):
        return {
            "query": kwargs["query"],
            "provider": "fake",
            "results": kwargs["results"][: kwargs.get("top_k") or len(kwargs["results"])],
            "warnings": [],
            "reranking": {"used": True, "path": "fake"},
        }


class FakeGraphRetrievalClient:
    async def retrieve(self, **kwargs):
        return {
            "query": kwargs["query"],
            "retrieval_endpoint": kwargs["retrieval_endpoint"],
            "context_blocks": [
                {
                    "chunk_id": "chunk-1",
                    "citation_label": "doc#chunk-1",
                    "content": "retrieved graph evidence",
                }
            ],
            "evidence_blocks": [],
            "grounded_user_message": "retrieved graph evidence",
            "warnings": [],
            "degraded": False,
        }


class FakeGraphRepoContextClient:
    async def explore_repository(self, **kwargs):
        return SimpleNamespace(
            text="Repo: repo\nCitations:\n- src/app.py:1\nAnswer:\nrepo evidence",
            json={
                "answer": "repo evidence",
                "query": kwargs["query"],
                "repo_root": "/tmp/repo",
                "citations": [{"path": "src/app.py", "start_line": 1}],
                "raw_locations": [
                    {
                        "path": "src/app.py",
                        "start_line": 1,
                        "end_line": 1,
                        "text": "repo evidence",
                        "truncated": False,
                    }
                ],
                "turns_used": 1,
                "truncated": False,
                "warnings": [],
            },
            metadata={"kind": "repo_context", "repo_name": kwargs["repo_name"]},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("graph_id", "graph_input"),
    [
        ("research_brief", {"question": "ship graph support"}),
        ("implementation_plan", {"goal": "ship graph support"}),
        ("threaded_search", {"latest_user_prompt": "ship graph support"}),
        ("threaded_rag", {"latest_user_prompt": "ship graph support"}),
        (
            "repo_context",
            {
                "latest_user_prompt": "ship graph support",
                "repo_name": "llm-control-plane",
            },
        ),
    ],
)
async def test_shipped_generated_graphs_execute_with_neutral_runtime_clients(
    tmp_path,
    monkeypatch,
    graph_id,
    graph_input,
):
    monkeypatch.setattr(graph_common, "LLM_CLIENT", FakeGraphLLMClient())
    monkeypatch.setattr(graph_common, "SEARCH_CLIENT", FakeGraphSearchClient())
    monkeypatch.setattr(graph_common, "RETRIEVAL_CLIENT", FakeGraphRetrievalClient())
    monkeypatch.setattr(graph_common, "REPO_CONTEXT_CLIENT", FakeGraphRepoContextClient())
    registry = GraphRegistry()
    registry.load()
    store = SQLiteGraphRunStore(tmp_path / "graphs.sqlite3")
    await store.initialize()
    try:
        executor = GraphExecutor(registry, store)
        created = await executor.create_run(
            graph_id,
            input=graph_input,
            config={
                "configurable": {
                    "thread_id": f"{graph_id}-thread",
                    "endpoint": "selected-model",
                    "retrieval_endpoint": "http://retrieval/api/retrieve/context",
                }
            },
        )

        snapshot = await executor.run(created["run_id"])

        assert snapshot["run"]["status"] == "completed"
        assert snapshot["run"]["output"]["final"].startswith("answer:")
        planner_key = {
            "threaded_search": "search_plan",
            "threaded_rag": "rag_prompt",
            "repo_context": "repo_context_plan",
        }.get(graph_id)
        if planner_key:
            outputs = snapshot["run"]["output"]["outputs"]
            assert outputs[planner_key]["metadata"]["endpoint"] is None
            assert outputs[planner_key]["metadata"]["workload_type"] == "utility"
            assert outputs["reply"]["metadata"]["endpoint"] == "selected-model"
    finally:
        await store.close()


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
