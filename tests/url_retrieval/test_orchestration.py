import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from src.dashboard.workflow_server_helpers import build_workflow_chat_run_payload, workflow_chat_response_text
from src.graphs import common
from src.orchestrator.graph.api import create_graph_router
from src.orchestrator.graph.executor import GraphExecutor
from src.orchestrator.graph.registry import GraphRegistry
from src.orchestrator.graph.store import SQLiteGraphRunStore
from src.orchestrator.workflow.api import create_workflow_router
from src.orchestrator.workflow.executor import WorkflowExecutor
from src.orchestrator.workflow.registry import WorkflowRegistry
from src.orchestrator.workflow.store import SQLiteWorkflowStore


class AnswerClient:
    answer = "- First finding.\n- Second finding."

    def __init__(self):
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return {"text": self.answer, "metadata": {"endpoint": kwargs["endpoint"]}}

    async def stream_complete(self, **kwargs):
        result = await self.complete(**kwargs)
        yield {"channel": "content", "content": result["text"], "metadata": result["metadata"]}


@pytest_asyncio.fixture
async def orchestration(tmp_path, monkeypatch):
    llm = AnswerClient()
    monkeypatch.setattr(common, "LLM_CLIENT", llm)
    workflows = WorkflowRegistry()
    workflows.load()
    graphs = GraphRegistry()
    graphs.load()
    workflow_store = SQLiteWorkflowStore(tmp_path / "workflows.sqlite3")
    graph_store = SQLiteGraphRunStore(tmp_path / "graphs.sqlite3")
    await workflow_store.initialize()
    await graph_store.initialize()
    workflow_executor = WorkflowExecutor(workflows, workflow_store, llm)
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
async def test_workflow_and_graph_answer_the_prompt_after_deterministic_retrieval(
    orchestration, web_backend, stream,
):
    app, workflow, graph = orchestration
    requests, _ = web_backend(lambda request: httpx.Response(
        200, text="Reference content", headers={"content-type": "text/plain"},
    ))
    params = {
        "latest_user_prompt": "Summarize https://example.com/page#one and https://example.com/page#two in two bullets.",
        "manual_source_text": "Keep the requested format.",
        "uploaded_source_text": "Supporting context.",
        "conversation_messages": [{"role": "user", "content": "Previous turn"}],
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        dispatch_payload = build_workflow_chat_run_payload(
            workflow.registry.get("url_retrieval").to_dict(),
            **params, endpoint="selected-model", retrieval_endpoint="http://unused-retrieval/context",
        )
        created_workflow = await client.post("/workflows/url_retrieval/runs", json=dispatch_payload)
        created_graph = await client.post("/graphs/url_retrieval/runs", json={
            "input": params, "config": {"configurable": {
                "endpoint": "selected-model", "retrieval_endpoint": "http://unused-retrieval/context",
            }},
        })
    assert created_workflow.status_code == created_graph.status_code == 200
    workflow_id = created_workflow.json()["run_id"]
    graph_id = created_graph.json()["run_id"]
    await workflow.advance(workflow_id)
    await workflow.advance(workflow_id)
    assert workflow.llm_client.calls == []
    if stream:
        workflow_events = [event async for event in workflow.run_to_completion_stream(workflow_id)]
        graph_events = [event async for event in graph.stream(graph_id)]
        assert any(event["type"] == "step_completed" for event in workflow_events)
        assert any(event["type"] == "step_delta" and event["content"] == AnswerClient.answer
                   for event in workflow_events)
        assert any("fetch_urls" in event.get("payload", {}) for event in graph_events)
    else:
        await workflow.run_to_completion(workflow_id)
        await graph.run(graph_id)
    workflow_snapshot = await workflow.store.snapshot(workflow_id)
    graph_snapshot = await graph.store.snapshot(graph_id)
    assert workflow_snapshot["run"]["status"] == graph_snapshot["run"]["status"] == "completed"
    assert workflow_chat_response_text(workflow_snapshot) == AnswerClient.answer
    workflow_output = workflow_snapshot["steps"][1]["output_json"]
    graph_state = graph_snapshot["run"]["output"]
    if stream:
        graph_state = graph_state["last_event"]["payload"]["answer_from_urls"]
    assert graph_state["final"] == AnswerClient.answer
    assert graph_state["outputs"]["reply"]["text"] == AnswerClient.answer
    graph_output = graph_state["outputs"]["url_documents"]
    assert workflow_output["text"] == graph_output["text"]
    assert workflow_output["metadata"] == graph_output["metadata"]
    for output in (workflow_output, graph_output):
        output["json"]["results"][0].pop("fetched_at")
    assert workflow_output["json"] == graph_output["json"]
    assert any(artifact["name"] == "url_documents" and artifact["content_text"] == workflow_output["text"]
               for artifact in workflow_snapshot["artifacts"])
    assert len(requests) == 2
    assert len(workflow.llm_client.calls) == 2
    for call in workflow.llm_client.calls:
        assert params["latest_user_prompt"] in call["prompt"]
        assert "Reference content" in call["prompt"]
        assert "requested output format" in call["prompt"]
        assert call["endpoint"] == "selected-model"
        assert call["retrieval_endpoint"] is None
        for key in ("manual_source_text", "uploaded_source_text", "conversation_messages"):
            assert call["context"][key] == params[key]


@pytest.mark.asyncio
async def test_no_url_runs_still_answer_the_prompt_without_network_retrieval(orchestration, web_backend):
    _, workflow, graph = orchestration
    requests, resolved = web_backend(lambda request: pytest.fail("unexpected request"))
    params = {"latest_user_prompt": "A prompt without any links"}
    wf = await workflow.create_run("url_retrieval", params=params, endpoint="selected-model")
    gr = await graph.create_run("url_retrieval", input=params, config={"configurable": {"endpoint": "selected-model"}})
    wf = await workflow.run_to_completion(wf["run"]["run_id"])
    gr = await graph.run(gr["run_id"])
    assert wf["run"]["status"] == gr["run"]["status"] == "completed"
    assert wf["steps"][1]["output_json"]["json"] == {"results": []}
    assert gr["run"]["output"]["outputs"]["url_documents"]["json"] == {"results": []}
    assert requests == resolved == []
    assert workflow_chat_response_text(wf) == gr["run"]["output"]["final"] == AnswerClient.answer
    assert len(workflow.llm_client.calls) == 2


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
        "url_retrieval", params={"latest_user_prompt": "https://example.com"}, endpoint="selected-model",
    )
    run_id = created["run"]["run_id"]
    failed_document = await workflow.run_to_completion(run_id)
    assert failed_document["run"]["status"] == "completed"
    assert failed_document["steps"][1]["output_json"]["metadata"]["failed"] == 1
    assert "503" in workflow.llm_client.calls[0]["prompt"]
    assert workflow_chat_response_text(failed_document) == AnswerClient.answer
    await workflow.retry_step(run_id, "fetch_urls")
    retried = await workflow.run_to_completion(run_id)
    assert retried["steps"][1]["output_json"]["metadata"]["successful"] == 1
    assert retried["steps"][0]["started_at"] == failed_document["steps"][0]["started_at"]
    assert "Recovered" in workflow.llm_client.calls[1]["prompt"]
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_unexpected_errors_fail_the_workflow_and_graph(orchestration, web_backend):
    _, workflow, graph = orchestration

    def bug(request):
        raise RuntimeError("implementation bug")

    web_backend(bug)
    params = {"latest_user_prompt": "https://example.com"}
    wf = await workflow.create_run("url_retrieval", params=params, endpoint="selected-model")
    gr = await graph.create_run("url_retrieval", input=params)
    wf = await workflow.run_to_completion(wf["run"]["run_id"])
    gr = await graph.run(gr["run_id"])
    assert wf["run"]["status"] == gr["run"]["status"] == "failed"
    assert "implementation bug" in wf["steps"][1]["error"]
    assert "implementation bug" in gr["run"]["error"]
    assert workflow.llm_client.calls == []
