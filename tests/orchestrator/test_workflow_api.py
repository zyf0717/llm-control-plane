from pathlib import Path

from fastapi.testclient import TestClient

from src.orchestrator import proxy as proxy_module
from src.orchestrator.proxy import app
from src.orchestrator.workflow_executor import WorkflowExecutor
from src.orchestrator.workflow_registry import WorkflowRegistry
from src.orchestrator.workflow_store import SQLiteWorkflowStore


class FakeLLMClient:
    async def complete(self, **kwargs):
        return {"text": "ok", "metadata": {"endpoint": kwargs["endpoint"]}}


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
steps:
  - id: first
    kind: llm
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
            json={"params": {"goal": "ship"}, "endpoint": "smart"},
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        advance_response = client.post(f"/workflow-runs/{run_id}/advance")
        assert advance_response.status_code == 200
        assert advance_response.json()["run"]["status"] == "completed"

        get_response = client.get(f"/workflow-runs/{run_id}")
        assert get_response.status_code == 200
        assert get_response.json()["steps"][0]["output_json"]["text"] == "ok"


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
