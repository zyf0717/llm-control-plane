from pathlib import Path
import json

from fastapi.testclient import TestClient

from src.orchestrator import proxy_services as proxy_module
from src.orchestrator.conversation_store import MemoryConversationStore
from src.orchestrator.proxy import app
from src.orchestrator.workflow.executor import WorkflowExecutor
from src.orchestrator.workflow.registry import WorkflowRegistry
from src.orchestrator.workflow.store import SQLiteWorkflowStore


class FakeLLMClient:
    async def complete(self, **kwargs):
        return {"text": "ok", "metadata": {"endpoint": kwargs["endpoint"]}}


class RecordingLLMClient(FakeLLMClient):
    def __init__(self):
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return await super().complete(**kwargs)


class FakeRepoContextListClient:
    def list_repositories(self):
        return ["alpha", "beta"]


def write_workflow(path: Path) -> None:
    path.write_text(
        """
id: sample
name: Sample
version: 0.1.0
params_schema:
  type: object
  required: [goal]
steps:
  - id: first
    kind: llm
    prompt: "{{ params.goal }}"
    output_key: first
""",
        encoding="utf-8",
    )


def write_visible_workflow(path: Path) -> None:
    path.write_text(
        """
id: visible
name: Visible
version: 0.1.0
params_schema:
  type: object
  required: [goal]
steps:
  - id: first
    kind: llm
    chat_visibility: final
    prompt: "{{ params.goal }}"
    output_key: first
""",
        encoding="utf-8",
    )


def test_workflow_api_create_advance_and_get(tmp_path):
    write_workflow(tmp_path / "sample.yaml")
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")

    with TestClient(app) as client:
        client.portal.call(store.initialize)
        executor = WorkflowExecutor(registry, store, FakeLLMClient())
        proxy_module.set_workflow_components(
            registry=registry,
            store=store,
            executor=executor,
        )

        list_response = client.get("/workflows")
        assert list_response.status_code == 200
        assert list_response.json()["workflows"][0]["id"] == "sample"

        create_response = client.post(
            "/workflows/sample/runs",
            json={
                "params": {"goal": "ship"},
                "endpoint": "node-a",
                "retrieval_endpoint": "http://retrieval/api/retrieve/context",
                "search_provider": "duckduckgo_html",
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]
        assert (
            create_response.json()["snapshot"]["run"]["retrieval_endpoint"]
            == "http://retrieval/api/retrieve/context"
        )
        assert (
            create_response.json()["snapshot"]["run"]["search_provider"]
            == "duckduckgo_html"
        )

        advance_response = client.post(f"/workflow-runs/{run_id}/advance")
        assert advance_response.status_code == 200
        assert advance_response.json()["run"]["status"] == "completed"

        get_response = client.get(f"/workflow-runs/{run_id}")
        assert get_response.status_code == 200
        assert get_response.json()["steps"][0]["output_json"]["text"] == "ok"


def test_workflow_llm_steps_skip_durable_conversation_history(tmp_path):
    write_workflow(tmp_path / "sample.yaml")
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    llm_client = RecordingLLMClient()

    with TestClient(app) as client:
        client.portal.call(store.initialize)
        proxy_module.set_workflow_components(
            registry=registry,
            store=store,
            executor=WorkflowExecutor(registry, store, llm_client),
        )

        create_response = client.post(
            "/workflows/sample/runs",
            json={
                "params": {"goal": "ship"},
                "endpoint": "node-a",
                "conversation_id": "dashboard-conversation",
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        advance_response = client.post(f"/workflow-runs/{run_id}/advance")

    assert advance_response.status_code == 200
    assert llm_client.calls
    assert all(call["skip_conversation"] is True for call in llm_client.calls)


def test_workflow_api_validates_required_params(tmp_path):
    write_workflow(tmp_path / "sample.yaml")
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")

    with TestClient(app) as client:
        client.portal.call(store.initialize)
        proxy_module.set_workflow_components(
            registry=registry,
            store=store,
            executor=WorkflowExecutor(registry, store, FakeLLMClient()),
        )

        response = client.post("/workflows/sample/runs", json={"params": {}})

        assert response.status_code == 400
        assert "missing required workflow params" in response.json()["detail"]


def test_workflow_api_run_stream_emits_ordered_events(tmp_path):
    write_visible_workflow(tmp_path / "visible.yaml")
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")

    with TestClient(app) as client:
        client.portal.call(store.initialize)
        proxy_module.set_workflow_components(
            registry=registry,
            store=store,
            executor=WorkflowExecutor(registry, store, FakeLLMClient()),
        )
        create_response = client.post(
            "/workflows/visible/runs",
            json={"params": {"goal": "ship"}, "endpoint": "node-a"},
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        with client.stream(
            "POST",
            f"/workflow-runs/{run_id}/run-stream",
            json={"stream": False},
        ) as response:
            assert response.status_code == 200
            events = _decode_sse_events(response.read().decode("utf-8"))

        event_types = [event["type"] for event in events]
        assert event_types[:3] == ["run_started", "snapshot", "step_started"]
        assert "step_completed" in event_types
        assert event_types[-1] == "run_completed"
        completed = next(event for event in events if event["type"] == "step_completed")
        assert completed["chat_visibility"] == "final"
        assert completed["content"] == "ok"
        assert events[-1]["snapshot"]["run"]["status"] == "completed"


def test_workflow_api_rejects_smart_endpoint(tmp_path):
    write_workflow(tmp_path / "sample.yaml")
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")

    with TestClient(app) as client:
        client.portal.call(store.initialize)
        proxy_module.set_workflow_components(
            registry=registry,
            store=store,
            executor=WorkflowExecutor(registry, store, FakeLLMClient()),
        )

        response = client.post(
            "/workflows/sample/runs",
            json={"params": {"goal": "ship"}, "endpoint": "smart"},
        )

        assert response.status_code == 400
        assert "concrete endpoint" in response.json()["detail"]


def test_workflow_api_delete_runs_clears_history(tmp_path):
    write_workflow(tmp_path / "sample.yaml")
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")

    with TestClient(app) as client:
        client.portal.call(store.initialize)
        proxy_module.set_workflow_components(
            registry=registry,
            store=store,
            executor=WorkflowExecutor(registry, store, FakeLLMClient()),
        )
        create_response = client.post(
            "/workflows/sample/runs",
            json={"params": {"goal": "ship"}, "endpoint": "node-a"},
        )
        assert create_response.status_code == 200

        delete_response = client.delete("/workflow-runs")

        assert delete_response.status_code == 200
        assert delete_response.json()["deleted"]["workflow_runs"] == 1
        assert client.get("/workflow-runs").json()["runs"] == []


def test_repo_context_repos_api_lists_directories(monkeypatch):
    monkeypatch.setattr(
        proxy_module,
        "repo_context_client",
        FakeRepoContextListClient(),
    )

    with TestClient(app) as client:
        response = client.get("/repo-context/repos")

    assert response.status_code == 200
    assert response.json() == {"repositories": ["alpha", "beta"]}


def test_workflow_api_thread_mode_enriches_params_and_keeps_conversation_separate(
    tmp_path,
):
    write_workflow(tmp_path / "sample.yaml")
    registry = WorkflowRegistry(tmp_path)
    registry.load()
    workflow_store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    conversation_store = MemoryConversationStore()

    async def upsert_thread_state():
        await conversation_store.upsert_thread_state(
            "main-thread",
            covered_message_id=records[0]["id"],
            state_text="prior thread state",
        )

    with TestClient(app) as client:
        client.portal.call(workflow_store.initialize)
        client.portal.call(
            conversation_store.append_messages,
            "main-thread",
            [
                {"role": "user", "content": "Prior user"},
                {"role": "assistant", "content": "Prior assistant"},
            ],
        )
        records = client.portal.call(
            conversation_store.get_conversation_message_records,
            "main-thread",
        )
        client.portal.call(upsert_thread_state)
        proxy_module.set_conversation_store(conversation_store)
        proxy_module.set_workflow_components(
            registry=registry,
            store=workflow_store,
            executor=WorkflowExecutor(registry, workflow_store, FakeLLMClient()),
        )

        response = client.post(
            "/workflows/sample/runs",
            json={
                "params": {"goal": "ship", "thread_briefing": "manual"},
                "endpoint": "node-a",
                "conversation_id": "workflow-internal",
                "source_conversation_id": "main-thread",
                "history_mode": "thread",
                "recent_message_count": 1,
            },
        )

    assert response.status_code == 200
    run = response.json()["snapshot"]["run"]
    assert run["conversation_id"] == "workflow-internal"
    assert run["params"]["thread_state_text"] == "prior thread state"
    assert "Prior assistant" in run["params"]["recent_conversation_messages"]
    assert run["params"]["thread_briefing"].startswith("manual")
    assert "source_conversation_id" in run["params"]["thread_metadata"]
    assert run["params"]["thread_metadata"]["source_conversation_id"] == "main-thread"


def _decode_sse_events(text: str) -> list[dict]:
    events = []
    for block in text.strip().split("\n\n"):
        data_lines = [
            line[len("data:") :].strip()
            for line in block.splitlines()
            if line.startswith("data:")
        ]
        if data_lines:
            events.append(json.loads("\n".join(data_lines)))
    return events
