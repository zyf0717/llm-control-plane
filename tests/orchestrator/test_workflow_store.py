import pytest

from src.orchestrator.workflow.models import WorkflowRun
from src.orchestrator.workflow.store import SQLiteWorkflowStore


@pytest.mark.asyncio
async def test_workflow_store_round_trip_snapshot(tmp_path):
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    try:
        run = WorkflowRun(
            run_id="wf_test",
            workflow_id="sample",
            workflow_version="0.1.0",
            status="pending",
            convo_id="wf_test_convo",
            params={"goal": "ship"},
            endpoint="smart",
            reasoning_effort="high",
            current_step_id=None,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            completed_at=None,
        )
        await store.create_run(run, ["first", "second"])
        await store.mark_step_running("wf_test", "first", {"params": run.params})
        await store.mark_step_completed(
            "wf_test",
            "first",
            {"text": "done", "json": None, "metadata": {}},
        )

        snapshot = await store.snapshot("wf_test")

        assert snapshot["run"]["run_id"] == "wf_test"
        assert snapshot["steps"][0]["status"] == "completed"
        assert snapshot["steps"][0]["output_json"]["text"] == "done"
        assert snapshot["steps"][1]["status"] == "pending"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_workflow_store_retry_requires_failed_step(tmp_path):
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    try:
        run = WorkflowRun(
            run_id="wf_test",
            workflow_id="sample",
            workflow_version="0.1.0",
            status="pending",
            convo_id="wf_test_convo",
            params={},
            endpoint=None,
            reasoning_effort=None,
            current_step_id=None,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            completed_at=None,
        )
        await store.create_run(run, ["first"])

        with pytest.raises(ValueError, match="only failed"):
            await store.retry_step("wf_test", "first")

        await store.mark_step_failed("wf_test", "first", "boom")
        await store.retry_step("wf_test", "first")

        snapshot = await store.snapshot("wf_test")
        assert snapshot["run"]["status"] == "running"
        assert snapshot["steps"][0]["status"] == "pending"
    finally:
        await store.close()
