from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from ..history_store import DEFAULT_HISTORY_DB_PATH
from .models import WorkflowRun, WorkflowRunStatus, WorkflowStepRun


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteWorkflowStore:
    backend_name = "sqlite"

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self._conn: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()

    async def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self.db_path))
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS workflow_runs (
                run_id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                workflow_version TEXT NOT NULL,
                status TEXT NOT NULL,
                convo_id TEXT NOT NULL,
                params_json TEXT NOT NULL,
                endpoint TEXT,
                reasoning_effort TEXT,
                rag_endpoint TEXT,
                search_provider TEXT,
                current_step_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            )
            """)
        await self._ensure_workflow_run_columns()
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_workflow_runs_status_updated
            ON workflow_runs (status, updated_at DESC)
            """)
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_workflow_runs_convo_id
            ON workflow_runs (convo_id)
            """)
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS workflow_step_runs (
                run_id TEXT NOT NULL,
                step_id TEXT NOT NULL,
                status TEXT NOT NULL,
                input_json TEXT NOT NULL DEFAULT '{}',
                output_json TEXT,
                error TEXT,
                started_at TEXT,
                completed_at TEXT,
                PRIMARY KEY (run_id, step_id),
                FOREIGN KEY (run_id) REFERENCES workflow_runs(run_id)
            )
            """)
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS workflow_artifacts (
                artifact_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                step_id TEXT,
                artifact_type TEXT NOT NULL,
                name TEXT NOT NULL,
                content_json TEXT,
                content_text TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES workflow_runs(run_id)
            )
            """)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def create_run(self, run: WorkflowRun, step_ids: list[str]) -> None:
        conn = self._require_connection()
        async with self._write_lock:
            await self._begin_immediate(conn)
            try:
                await conn.execute(
                    """
                    INSERT INTO workflow_runs (
                        run_id, workflow_id, workflow_version, status, convo_id,
                        params_json, endpoint, reasoning_effort, rag_endpoint,
                        search_provider, current_step_id, created_at, updated_at,
                        completed_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run.run_id,
                        run.workflow_id,
                        run.workflow_version,
                        run.status,
                        run.convo_id,
                        json.dumps(run.params),
                        run.endpoint,
                        run.reasoning_effort,
                        run.rag_endpoint,
                        run.search_provider,
                        run.current_step_id,
                        run.created_at,
                        run.updated_at,
                        run.completed_at,
                    ),
                )
                await conn.executemany(
                    """
                    INSERT INTO workflow_step_runs (
                        run_id, step_id, status, input_json,
                        output_json, error, started_at, completed_at
                    )
                    VALUES (?, ?, 'pending', '{}', NULL, NULL, NULL, NULL)
                    """,
                    [(run.run_id, step_id) for step_id in step_ids],
                )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    async def get_run(self, run_id: str) -> WorkflowRun | None:
        conn = self._require_connection()
        cursor = await conn.execute(
            """
            SELECT run_id, workflow_id, workflow_version, status, convo_id,
                   params_json, endpoint, reasoning_effort, rag_endpoint,
                   search_provider, current_step_id, created_at, updated_at,
                   completed_at
            FROM workflow_runs
            WHERE run_id = ?
            """,
            (run_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return self._row_to_run(row) if row else None

    async def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        conn = self._require_connection()
        cursor = await conn.execute(
            """
            SELECT run_id, workflow_id, workflow_version, status, convo_id,
                   params_json, endpoint, reasoning_effort, rag_endpoint,
                   search_provider, current_step_id, created_at, updated_at,
                   completed_at
            FROM workflow_runs
            ORDER BY updated_at DESC, run_id ASC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [self._row_to_run(row).to_dict() for row in rows]

    async def clear_runs(self) -> dict[str, int]:
        conn = self._require_connection()
        async with self._write_lock:
            await self._begin_immediate(conn)
            try:
                artifact_count = await self._table_count(conn, "workflow_artifacts")
                step_count = await self._table_count(conn, "workflow_step_runs")
                run_count = await self._table_count(conn, "workflow_runs")
                await conn.execute("DELETE FROM workflow_artifacts")
                await conn.execute("DELETE FROM workflow_step_runs")
                await conn.execute("DELETE FROM workflow_runs")
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise
        return {
            "workflow_artifacts": artifact_count,
            "workflow_step_runs": step_count,
            "workflow_runs": run_count,
        }

    async def get_step_runs(self, run_id: str) -> list[WorkflowStepRun]:
        conn = self._require_connection()
        cursor = await conn.execute(
            """
            SELECT run_id, step_id, status, input_json, output_json, error,
                   started_at, completed_at
            FROM workflow_step_runs
            WHERE run_id = ?
            ORDER BY rowid ASC
            """,
            (run_id,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [self._row_to_step(row) for row in rows]

    async def snapshot(self, run_id: str) -> dict[str, Any] | None:
        run = await self.get_run(run_id)
        if run is None:
            return None
        return {
            "run": run.to_dict(),
            "steps": [step.to_dict() for step in await self.get_step_runs(run_id)],
            "artifacts": await self.list_artifacts(run_id),
        }

    async def mark_run_status(
        self,
        run_id: str,
        status: WorkflowRunStatus,
        *,
        current_step_id: str | None = None,
        completed: bool = False,
    ) -> None:
        conn = self._require_connection()
        now = _utc_now()
        completed_at = now if completed else None
        async with self._write_lock:
            await self._begin_immediate(conn)
            try:
                await conn.execute(
                    """
                    UPDATE workflow_runs
                    SET status = ?, current_step_id = ?, updated_at = ?,
                        completed_at = COALESCE(?, completed_at)
                    WHERE run_id = ?
                    """,
                    (status, current_step_id, now, completed_at, run_id),
                )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    async def mark_step_running(
        self, run_id: str, step_id: str, input_json: dict[str, Any]
    ) -> None:
        conn = self._require_connection()
        now = _utc_now()
        async with self._write_lock:
            await self._begin_immediate(conn)
            try:
                await conn.execute(
                    """
                    UPDATE workflow_step_runs
                    SET status = 'running', input_json = ?, error = NULL,
                        started_at = ?, completed_at = NULL
                    WHERE run_id = ? AND step_id = ?
                    """,
                    (json.dumps(input_json), now, run_id, step_id),
                )
                await conn.execute(
                    """
                    UPDATE workflow_runs
                    SET status = 'running', current_step_id = ?, updated_at = ?
                    WHERE run_id = ?
                    """,
                    (step_id, now, run_id),
                )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    async def mark_step_completed(
        self, run_id: str, step_id: str, output_json: dict[str, Any]
    ) -> None:
        conn = self._require_connection()
        now = _utc_now()
        async with self._write_lock:
            await self._begin_immediate(conn)
            try:
                await conn.execute(
                    """
                    UPDATE workflow_step_runs
                    SET status = 'completed', output_json = ?, error = NULL,
                        completed_at = ?
                    WHERE run_id = ? AND step_id = ?
                    """,
                    (json.dumps(output_json), now, run_id, step_id),
                )
                await conn.execute(
                    """
                    UPDATE workflow_runs
                    SET status = 'running', current_step_id = NULL, updated_at = ?
                    WHERE run_id = ?
                    """,
                    (now, run_id),
                )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    async def mark_step_failed(self, run_id: str, step_id: str, error: str) -> None:
        conn = self._require_connection()
        now = _utc_now()
        async with self._write_lock:
            await self._begin_immediate(conn)
            try:
                await conn.execute(
                    """
                    UPDATE workflow_step_runs
                    SET status = 'failed', error = ?, completed_at = ?
                    WHERE run_id = ? AND step_id = ?
                    """,
                    (str(error), now, run_id, step_id),
                )
                await conn.execute(
                    """
                    UPDATE workflow_runs
                    SET status = 'failed', current_step_id = ?, updated_at = ?,
                        completed_at = ?
                    WHERE run_id = ?
                    """,
                    (step_id, now, now, run_id),
                )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    async def retry_step(
        self, run_id: str, step_id: str, reset_step_ids: list[str] | None = None
    ) -> None:
        conn = self._require_connection()
        now = _utc_now()
        reset_ids = list(dict.fromkeys([*(reset_step_ids or []), step_id]))
        async with self._write_lock:
            await self._begin_immediate(conn)
            try:
                cursor = await conn.execute(
                    """
                    SELECT status FROM workflow_step_runs
                    WHERE run_id = ? AND step_id = ?
                    """,
                    (run_id, step_id),
                )
                row = await cursor.fetchone()
                await cursor.close()
                if not row:
                    raise KeyError(f"unknown workflow step run: {run_id}/{step_id}")
                if row[0] not in {"failed", "completed"}:
                    raise ValueError(
                        "only failed or completed workflow steps can be retried"
                    )

                placeholders = ", ".join("?" for _ in reset_ids)
                cursor = await conn.execute(
                    f"""
                    SELECT step_id, status FROM workflow_step_runs
                    WHERE run_id = ? AND step_id IN ({placeholders})
                    """,
                    (run_id, *reset_ids),
                )
                reset_rows = await cursor.fetchall()
                await cursor.close()
                running_steps = [
                    row[0] for row in reset_rows if row[1] == "running"
                ]
                if running_steps:
                    raise ValueError(
                        "running workflow steps cannot be reset: "
                        + ", ".join(str(step_id) for step_id in running_steps)
                    )

                await conn.execute(
                    f"""
                    UPDATE workflow_step_runs
                    SET status = 'pending', output_json = NULL, error = NULL,
                        started_at = NULL, completed_at = NULL
                    WHERE run_id = ? AND step_id IN ({placeholders})
                    """,
                    (run_id, *reset_ids),
                )
                await conn.execute(
                    f"""
                    DELETE FROM workflow_artifacts
                    WHERE run_id = ? AND step_id IN ({placeholders})
                    """,
                    (run_id, *reset_ids),
                )
                await conn.execute(
                    """
                    UPDATE workflow_runs
                    SET status = 'running', current_step_id = NULL, updated_at = ?,
                        completed_at = NULL
                    WHERE run_id = ?
                    """,
                    (now, run_id),
                )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    async def create_artifact(
        self,
        *,
        artifact_id: str,
        run_id: str,
        step_id: str | None,
        artifact_type: str,
        name: str,
        content_json: dict[str, Any] | None = None,
        content_text: str | None = None,
    ) -> None:
        conn = self._require_connection()
        async with self._write_lock:
            await self._begin_immediate(conn)
            try:
                await conn.execute(
                    """
                    INSERT INTO workflow_artifacts (
                        artifact_id, run_id, step_id, artifact_type, name,
                        content_json, content_text, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        artifact_id,
                        run_id,
                        step_id,
                        artifact_type,
                        name,
                        json.dumps(content_json) if content_json is not None else None,
                        content_text,
                        _utc_now(),
                    ),
                )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    async def list_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        conn = self._require_connection()
        cursor = await conn.execute(
            """
            SELECT artifact_id, run_id, step_id, artifact_type, name, content_json,
                   content_text, created_at
            FROM workflow_artifacts
            WHERE run_id = ?
            ORDER BY created_at ASC, artifact_id ASC
            """,
            (run_id,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        artifacts = []
        for row in rows:
            artifacts.append(
                {
                    "artifact_id": row[0],
                    "run_id": row[1],
                    "step_id": row[2],
                    "artifact_type": row[3],
                    "name": row[4],
                    "content_json": _loads_json(row[5]) if row[5] else None,
                    "content_text": row[6],
                    "created_at": row[7],
                }
            )
        return artifacts

    async def _begin_immediate(self, conn: aiosqlite.Connection) -> None:
        if conn.in_transaction:
            await conn.rollback()
        await conn.execute("BEGIN IMMEDIATE")

    async def _ensure_workflow_run_columns(self) -> None:
        conn = self._require_connection()
        cursor = await conn.execute("PRAGMA table_info(workflow_runs)")
        rows = await cursor.fetchall()
        await cursor.close()
        existing = {str(row[1]) for row in rows}
        for column in ["rag_endpoint", "search_provider"]:
            if column not in existing:
                await conn.execute(f"ALTER TABLE workflow_runs ADD COLUMN {column} TEXT")

    @staticmethod
    async def _table_count(conn: aiosqlite.Connection, table: str) -> int:
        cursor = await conn.execute(f"SELECT COUNT(*) FROM {table}")
        row = await cursor.fetchone()
        await cursor.close()
        return int(row[0] or 0)

    def _require_connection(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("SQLiteWorkflowStore is not initialized")
        return self._conn

    @staticmethod
    def _row_to_run(row: Any) -> WorkflowRun:
        return WorkflowRun(
            run_id=row[0],
            workflow_id=row[1],
            workflow_version=row[2],
            status=row[3],
            convo_id=row[4],
            params=_loads_json(row[5]),
            endpoint=row[6],
            reasoning_effort=row[7],
            rag_endpoint=row[8],
            search_provider=row[9],
            current_step_id=row[10],
            created_at=row[11],
            updated_at=row[12],
            completed_at=row[13],
        )

    @staticmethod
    def _row_to_step(row: Any) -> WorkflowStepRun:
        return WorkflowStepRun(
            run_id=row[0],
            step_id=row[1],
            status=row[2],
            input_json=_loads_json(row[3]),
            output_json=_loads_json(row[4]) if row[4] else None,
            error=row[5],
            started_at=row[6],
            completed_at=row[7],
        )


def build_workflow_store_from_env() -> SQLiteWorkflowStore:
    raw_db_path = os.getenv("WORKFLOW_DB_PATH") or os.getenv("HISTORY_DB_PATH")
    db_path = Path(raw_db_path) if raw_db_path else DEFAULT_HISTORY_DB_PATH
    return SQLiteWorkflowStore(db_path)


def _loads_json(raw: str | bytes | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}
