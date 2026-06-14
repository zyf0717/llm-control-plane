import sqlite3

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
            conversation_id="wf_test_conversation",
            params={"goal": "ship"},
            endpoint="node-a",
            reasoning_effort="high",
            retrieval_endpoint="http://retrieval/api/retrieve/context",
            search_provider="duckduckgo_html",
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
        assert snapshot["run"]["retrieval_endpoint"] == "http://retrieval/api/retrieve/context"
        assert snapshot["run"]["search_provider"] == "duckduckgo_html"
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
            conversation_id="wf_test_conversation",
            params={},
            endpoint=None,
            reasoning_effort=None,
            retrieval_endpoint=None,
            search_provider=None,
            current_step_id=None,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            completed_at=None,
        )
        await store.create_run(run, ["first"])

        with pytest.raises(ValueError, match="only failed or completed"):
            await store.retry_step("wf_test", "first")

        await store.mark_step_failed("wf_test", "first", "boom")
        await store.retry_step("wf_test", "first")

        snapshot = await store.snapshot("wf_test")
        assert snapshot["run"]["status"] == "running"
        assert snapshot["steps"][0]["status"] == "pending"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_workflow_store_adds_current_run_columns(tmp_path):
    db_path = tmp_path / "workflow.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE workflow_runs (
            run_id TEXT PRIMARY KEY,
            workflow_id TEXT NOT NULL,
            workflow_version TEXT NOT NULL,
            status TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            params_json TEXT NOT NULL,
            endpoint TEXT,
            reasoning_effort TEXT,
            current_step_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()

    store = SQLiteWorkflowStore(db_path)
    await store.initialize()
    try:
        cursor = await store._require_connection().execute(
            "PRAGMA table_info(workflow_runs)"
        )
        columns = {row[1] for row in await cursor.fetchall()}
        await cursor.close()

        assert "retrieval_endpoint" in columns
        assert "search_provider" in columns
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_workflow_store_drops_legacy_run_schema(tmp_path):
    db_path = tmp_path / "workflow.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE workflow_runs (
            run_id TEXT PRIMARY KEY,
            workflow_id TEXT NOT NULL,
            workflow_version TEXT NOT NULL,
            status TEXT NOT NULL,
            convo_id TEXT NOT NULL,
            params_json TEXT NOT NULL,
            endpoint TEXT,
            reasoning_effort TEXT,
            rag_endpoint TEXT,
            current_step_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE workflow_step_runs (
            run_id TEXT NOT NULL,
            step_id TEXT NOT NULL,
            status TEXT NOT NULL,
            input_json TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY (run_id, step_id)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO workflow_runs (
            run_id, workflow_id, workflow_version, status, convo_id,
            params_json, created_at, updated_at
        ) VALUES (
            'wf_old', 'old', '0.1.0', 'pending', 'conversation-old',
            '{}', '2026-01-01', '2026-01-01'
        )
        """
    )
    conn.execute(
        """
        INSERT INTO workflow_step_runs (run_id, step_id, status)
        VALUES ('wf_old', 'step', 'pending')
        """
    )
    conn.commit()
    conn.close()

    store = SQLiteWorkflowStore(db_path)
    await store.initialize()
    try:
        cursor = await store._require_connection().execute(
            "PRAGMA table_info(workflow_runs)"
        )
        columns = {row[1] for row in await cursor.fetchall()}
        await cursor.close()

        assert "conversation_id" in columns
        assert "convo_id" not in columns
        assert "retrieval_endpoint" in columns
        assert "rag_endpoint" not in columns
        assert await store.list_runs() == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_workflow_store_clear_runs_deletes_runs_steps_and_artifacts(tmp_path):
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    try:
        run = WorkflowRun(
            run_id="wf_test",
            workflow_id="sample",
            workflow_version="0.1.0",
            status="pending",
            conversation_id="wf_test_conversation",
            params={},
            endpoint="node-a",
            reasoning_effort=None,
            retrieval_endpoint=None,
            search_provider=None,
            current_step_id=None,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            completed_at=None,
        )
        await store.create_run(run, ["first"])
        await store.create_artifact(
            artifact_id="artifact",
            run_id="wf_test",
            step_id="first",
            artifact_type="text",
            name="first",
            content_text="done",
        )

        deleted = await store.clear_runs()

        assert deleted == {
            "workflow_artifacts": 1,
            "workflow_step_runs": 1,
            "workflow_runs": 1,
        }
        assert await store.list_runs() == []
    finally:
        await store.close()
