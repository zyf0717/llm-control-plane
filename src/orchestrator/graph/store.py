from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

DEFAULT_GRAPH_DB_PATH = Path("var/graphs.sqlite3")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteGraphRunStore:
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
            CREATE TABLE IF NOT EXISTS graph_runs (
                run_id TEXT PRIMARY KEY,
                graph_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                status TEXT NOT NULL,
                input_json TEXT NOT NULL,
                config_json TEXT NOT NULL,
                output_json TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            )
            """)
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_graph_runs_status_updated
            ON graph_runs (status, updated_at DESC)
            """)
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_graph_runs_graph_updated
            ON graph_runs (graph_id, updated_at DESC)
            """)
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS graph_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                node_name TEXT,
                event_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES graph_runs(run_id)
            )
            """)
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS graph_artifacts (
                artifact_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                node_name TEXT,
                artifact_type TEXT NOT NULL,
                name TEXT NOT NULL,
                content_json TEXT,
                content_text TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES graph_runs(run_id)
            )
            """)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def create_run(
        self,
        *,
        run_id: str,
        graph_id: str,
        thread_id: str,
        input_json: dict[str, Any],
        config_json: dict[str, Any],
    ) -> None:
        conn = self._require_connection()
        now = _utc_now()
        async with self._write_lock:
            await self._begin_immediate(conn)
            try:
                await conn.execute(
                    """
                    INSERT INTO graph_runs (
                        run_id, graph_id, thread_id, status, input_json, config_json,
                        output_json, error, created_at, updated_at, completed_at
                    )
                    VALUES (?, ?, ?, 'pending', ?, ?, NULL, NULL, ?, ?, NULL)
                    """,
                    (
                        run_id,
                        graph_id,
                        thread_id,
                        json.dumps(input_json),
                        json.dumps(config_json),
                        now,
                        now,
                    ),
                )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        conn = self._require_connection()
        cursor = await conn.execute(
            """
            SELECT run_id, graph_id, thread_id, status, input_json, config_json,
                   output_json, error, created_at, updated_at, completed_at
            FROM graph_runs
            WHERE run_id = ?
            """,
            (run_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return _row_to_run(row) if row else None

    async def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        conn = self._require_connection()
        cursor = await conn.execute(
            """
            SELECT run_id, graph_id, thread_id, status, input_json, config_json,
                   output_json, error, created_at, updated_at, completed_at
            FROM graph_runs
            ORDER BY updated_at DESC, run_id ASC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [_row_to_run(row) for row in rows]

    async def snapshot(self, run_id: str) -> dict[str, Any] | None:
        run = await self.get_run(run_id)
        if run is None:
            return None
        return {
            "run": run,
            "events": await self.list_events(run_id),
            "artifacts": await self.list_artifacts(run_id),
        }

    async def mark_status(
        self,
        run_id: str,
        status: str,
        *,
        output_json: dict[str, Any] | None = None,
        error: str | None = None,
        completed: bool = False,
    ) -> None:
        conn = self._require_connection()
        now = _utc_now()
        completed_at = now if completed else None
        output_text = json.dumps(output_json) if output_json is not None else None
        async with self._write_lock:
            await self._begin_immediate(conn)
            try:
                await conn.execute(
                    """
                    UPDATE graph_runs
                    SET status = ?,
                        output_json = COALESCE(?, output_json),
                        error = ?,
                        updated_at = ?,
                        completed_at = COALESCE(?, completed_at)
                    WHERE run_id = ?
                    """,
                    (status, output_text, error, now, completed_at, run_id),
                )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    async def append_event(
        self,
        run_id: str,
        *,
        event_type: str,
        event_json: dict[str, Any],
        node_name: str | None = None,
    ) -> None:
        conn = self._require_connection()
        now = _utc_now()
        async with self._write_lock:
            await self._begin_immediate(conn)
            try:
                await conn.execute(
                    """
                    INSERT INTO graph_events (
                        run_id, event_type, node_name, event_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        event_type,
                        node_name,
                        json.dumps(event_json, default=str),
                        now,
                    ),
                )
                await conn.execute(
                    "UPDATE graph_runs SET updated_at = ? WHERE run_id = ?",
                    (now, run_id),
                )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    async def list_events(self, run_id: str, limit: int = 200) -> list[dict[str, Any]]:
        conn = self._require_connection()
        cursor = await conn.execute(
            """
            SELECT event_id, run_id, event_type, node_name, event_json, created_at
            FROM graph_events
            WHERE run_id = ?
            ORDER BY event_id ASC
            LIMIT ?
            """,
            (run_id, max(1, int(limit))),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [_row_to_event(row) for row in rows]

    async def list_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        conn = self._require_connection()
        cursor = await conn.execute(
            """
            SELECT artifact_id, run_id, node_name, artifact_type, name, content_json,
                   content_text, created_at
            FROM graph_artifacts
            WHERE run_id = ?
            ORDER BY created_at ASC, artifact_id ASC
            """,
            (run_id,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            {
                "artifact_id": row[0],
                "run_id": row[1],
                "node_name": row[2],
                "artifact_type": row[3],
                "name": row[4],
                "content_json": _loads_json(row[5]) if row[5] else None,
                "content_text": row[6],
                "created_at": row[7],
            }
            for row in rows
        ]

    def _require_connection(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Graph run store is not initialized")
        return self._conn

    @staticmethod
    async def _begin_immediate(conn: aiosqlite.Connection) -> None:
        await conn.execute("BEGIN IMMEDIATE")


def build_graph_store_from_env() -> SQLiteGraphRunStore:
    raw_db_path = os.getenv("GRAPH_DB_PATH")
    db_path = Path(raw_db_path) if raw_db_path else DEFAULT_GRAPH_DB_PATH
    return SQLiteGraphRunStore(db_path)


def _row_to_run(row: Any) -> dict[str, Any]:
    return {
        "run_id": row[0],
        "graph_id": row[1],
        "thread_id": row[2],
        "status": row[3],
        "input": _loads_json(row[4]),
        "config": _loads_json(row[5]),
        "output": _loads_json(row[6]) if row[6] else None,
        "error": row[7],
        "created_at": row[8],
        "updated_at": row[9],
        "completed_at": row[10],
    }


def _row_to_event(row: Any) -> dict[str, Any]:
    return {
        "event_id": row[0],
        "run_id": row[1],
        "event_type": row[2],
        "node_name": row[3],
        "event": _loads_json(row[4]),
        "created_at": row[5],
    }


def _loads_json(raw: str | bytes | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {"value": data}

