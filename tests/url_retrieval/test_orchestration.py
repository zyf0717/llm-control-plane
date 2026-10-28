import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from src.orchestrator.graph.api import create_graph_router
from src.orchestrator.graph.executor import GraphExecutor
from src.orchestrator.graph.registry import GraphRegistry
from src.orchestrator.graph.store import SQLiteGraphRunStore
from src.orchestrator.workflow.api import create_workflow_router
from src.orchestrator.workflow.executor import WorkflowExecutor
from src.orchestrator.workflow.registry import WorkflowRegistry
from src.orchestrator.workflow.store import SQLiteWorkflowStore


class NoLLMClient:
    async def complete(self, **kwargs):
        pytest.fail("URL retrieval must not call an LLM")

    async def stream_complete(self, **kwargs):
        pytest.fail("URL retrieval must not stream an LLM")
        yield


@pytest_asyncio.fixture
async def orchestration(tmp_path):
    workflows = WorkflowRegistry()
    workflows.load()
    graphs = GraphRegistry()
    graphs.load()
    workflow_store = SQLiteWorkflowStore(tmp_path / "workflows.sqlite3")
    graph_store = SQLiteGraphRunStore(tmp_path / "graphs.sqlite3")
    await workflow_store.initialize()
    await graph_store.initialize()
    workflow_executor = WorkflowExecutor(workflows, workflow_store, NoLLMClient())
    graph_executor = GraphExecutor(graphs, graph_store)
    app = FastAPI()
    app.include_router(create_workflow_router(
        registry_getter=lambda: workflows,
        store_getter=lambda: workflow_store,
        executor_getter=lambda: workflow_executor,
    ))
    app.include_router(create_graph_router(
        registry_getter=lambda: graphs,
        store_getter=lambda: graph_store,
        executor_getter=lambda: graph_executor,
    ))
    try:
        yield app, workflow_executor, graph_executor
    finally:
        await workflow_store.close()
        await graph_store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_workflow_and_graph_return_equivalent_documents_and_artifacts(
    orchestration, web_backend, stream, monkeypatch,
):
    from src.graphs import common

    monkeypatch.setattr(common, "LLM_CLIENT", NoLLMClient())
    app, workflow, graph = orchestration
    requests, _ = web_backend(lambda request: httpx.Response(
        200, text="Reference content", headers={"content-type": "text/plain"},
    ))
    params = {"latest_user_prompt": "Read https://example.com/page#one and https://example.com/page#two"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        created_workflow = await client.post("/workflows/url_retrieval/runs", json={"params": params})
        created_graph = await client.post("/graphs/url_retrieval/runs", json={"input": params})
    assert created_workflow.status_code == created_graph.status_code == 200
    workflow_id = created_workflow.json()["run_id"]
    graph_id = created_graph.json()["run_id"]
    if stream:
        workflow_events = [event async for event in workflow.run_to_completion_stream(workflow_id)]
        graph_events = [event async for event in graph.stream(graph_id)]
        assert any(event["type"] == "step_completed" for event in workflow_events)
        assert any("fetch_urls" in event.get("payload", {}) for event in graph_events)
    else:
        await workflow.run_to_completion(workflow_id)
        await graph.run(graph_id)
    workflow_snapshot = await workflow.store.snapshot(workflow_id)
    graph_snapshot = await graph.store.snapshot(graph_id)
    assert workflow_snapshot["run"]["status"] == graph_snapshot["run"]["status"] == "completed"
    assert workflow_snapshot["run"]["endpoint"] is None
    workflow_output = workflow_snapshot["steps"][1]["output_json"]
    graph_state = graph_snapshot["run"]["output"]
    if stream:
        graph_state = graph_state["last_event"]["payload"]["fetch_urls"]
    graph_output = graph_state["outputs"]["url_documents"]
    assert workflow_output["text"] == graph_output["text"]
    assert workflow_output["metadata"] == graph_output["metadata"]
    for output in (workflow_output, graph_output):
        output["json"]["results"][0].pop("fetched_at")
    assert workflow_output["json"] == graph_output["json"]
    assert workflow_snapshot["artifacts"][0]["content_text"] == workflow_output["text"]
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_no_url_runs_complete_without_network_or_model(orchestration, web_backend):
    _, workflow, graph = orchestration
    requests, resolved = web_backend(lambda request: pytest.fail("unexpected request"))
    params = {"latest_user_prompt": "A prompt without any links"}
    wf = await workflow.create_run("url_retrieval", params=params)
    gr = await graph.create_run("url_retrieval", input=params)
    wf = await workflow.run_to_completion(wf["run"]["run_id"])
    gr = await graph.run(gr["run_id"])
    assert wf["run"]["status"] == gr["run"]["status"] == "completed"
    assert wf["steps"][1]["output_json"]["json"] == {"results": []}
    assert gr["run"]["output"]["outputs"]["url_documents"]["json"] == {"results": []}
    assert requests == resolved == []


@pytest.mark.asyncio
async def test_all_url_failures_remain_inspectable_and_fetch_step_can_be_retried(orchestration, web_backend):
    _, workflow, _ = orchestration

    def handler(request):
        return httpx.Response(
            503 if len(requests) == 1 else 200,
            text="Recovered", headers={"content-type": "text/plain"},
        )

    requests, _ = web_backend(handler)
    created = await workflow.create_run(
        "url_retrieval", params={"latest_user_prompt": "https://example.com"},
    )
    run_id = created["run"]["run_id"]
    failed_document = await workflow.run_to_completion(run_id)
    assert failed_document["run"]["status"] == "completed"
    assert failed_document["steps"][1]["output_json"]["metadata"]["failed"] == 1
    await workflow.retry_step(run_id, "fetch_urls")
    retried = await workflow.run_to_completion(run_id)
    assert retried["steps"][1]["output_json"]["metadata"]["successful"] == 1
    assert retried["steps"][0]["started_at"] == failed_document["steps"][0]["started_at"]
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_unexpected_errors_fail_the_workflow_and_graph(orchestration, web_backend):
    _, workflow, graph = orchestration

    def bug(request):
        raise RuntimeError("implementation bug")

    web_backend(bug)
    params = {"latest_user_prompt": "https://example.com"}
    wf = await workflow.create_run("url_retrieval", params=params)
    gr = await graph.create_run("url_retrieval", input=params)
    wf = await workflow.run_to_completion(wf["run"]["run_id"])
    gr = await graph.run(gr["run_id"])
    assert wf["run"]["status"] == gr["run"]["status"] == "failed"
    assert "implementation bug" in wf["steps"][1]["error"]
    assert "implementation bug" in gr["run"]["error"]
