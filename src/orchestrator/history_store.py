import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import aiosqlite

logger = logging.getLogger(__name__)

DEFAULT_HISTORY_DB_PATH = Path("var/history.sqlite3")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_state(convo_id: str) -> Dict[str, object]:
    return {
        "convo_id": convo_id,
        "route_endpoint": None,
        "reasoning_effort": None,
        "slots": {},
        "updated_at": None,
    }


def _normalize_state(row: Optional[Dict[str, object]], convo_id: str) -> Dict[str, object]:
    state = _empty_state(convo_id)
    if not row:
        return state

    state.update(row)
    slots = state.get("slots")
    state["slots"] = deepcopy(slots) if isinstance(slots, dict) else {}
    return state


class HistoryStore(ABC):
    backend_name = "unknown"

    @abstractmethod
    async def initialize(self) -> None:
        """Prepare the backing store for use."""

    @abstractmethod
    async def close(self) -> None:
        """Release any backing-store resources."""

    @abstractmethod
    async def get_conversation(self, convo_id: str) -> Optional[List[Dict]]:
        """Return the stored conversation or None when it does not exist."""

    @abstractmethod
    async def append_messages(self, convo_id: str, messages: List[Dict]) -> None:
        """Append messages to a stored conversation."""

    @abstractmethod
    async def list_conversations(self) -> List[Dict[str, object]]:
        """Return conversation metadata sorted by most recently updated first."""

    @abstractmethod
    async def get_conversation_state(self, convo_id: str) -> Dict[str, object]:
        """Return route/reasoning/slot metadata for a conversation."""

    @abstractmethod
    async def update_conversation_state(
        self,
        convo_id: str,
        *,
        route_endpoint: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        valid_route_endpoints: Optional[List[str]] = None,
        clear_route: bool = False,
        allow_route_switch: bool = False,
        allow_reasoning_switch: bool = False,
    ) -> Dict[str, object]:
        """Atomically pin compatible route/reasoning metadata."""

    @abstractmethod
    async def set_conversation_slot(
        self, convo_id: str, endpoint: str, slot_id: int
    ) -> Dict[str, object]:
        """Persist a best-effort upstream slot mapping."""


class MemoryHistoryStore(HistoryStore):
    backend_name = "memory"

    def __init__(self):
        self.conversations: Dict[str, List[Dict]] = {}
        self.updated_at: Dict[str, str] = {}
        self.conversation_states: Dict[str, Dict[str, object]] = {}
        self._state_lock = asyncio.Lock()

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def get_conversation(self, convo_id: str) -> Optional[List[Dict]]:
        messages = self.conversations.get(convo_id)
        return deepcopy(messages) if messages is not None else None

    async def append_messages(self, convo_id: str, messages: List[Dict]) -> None:
        if not messages:
            return
        if convo_id not in self.conversations:
            self.conversations[convo_id] = []
        self.conversations[convo_id].extend(deepcopy(messages))
        self.updated_at[convo_id] = _utc_now()

    async def list_conversations(self) -> List[Dict[str, object]]:
        conversations = []
        for convo_id, messages in self.conversations.items():
            last_updated = self.updated_at.get(convo_id)
            if not last_updated:
                continue
            conversations.append(
                {
                    "convo_id": convo_id,
                    "last_updated": last_updated,
                    "message_count": len(messages),
                }
            )

        return sorted(
            conversations,
            key=lambda conversation: str(conversation["last_updated"]),
            reverse=True,
        )

    async def get_conversation_state(self, convo_id: str) -> Dict[str, object]:
        async with self._state_lock:
            return _normalize_state(self.conversation_states.get(convo_id), convo_id)

    async def update_conversation_state(
        self,
        convo_id: str,
        *,
        route_endpoint: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        valid_route_endpoints: Optional[List[str]] = None,
        clear_route: bool = False,
        allow_route_switch: bool = False,
        allow_reasoning_switch: bool = False,
    ) -> Dict[str, object]:
        async with self._state_lock:
            state = _normalize_state(self.conversation_states.get(convo_id), convo_id)
            result = self._apply_state_update(
                state,
                route_endpoint=route_endpoint,
                reasoning_effort=reasoning_effort,
                valid_route_endpoints=valid_route_endpoints,
                clear_route=clear_route,
                allow_route_switch=allow_route_switch,
                allow_reasoning_switch=allow_reasoning_switch,
            )
            if not result.get("conflict"):
                self.conversation_states[convo_id] = deepcopy(result["state"])
            return result

    async def set_conversation_slot(
        self, convo_id: str, endpoint: str, slot_id: int
    ) -> Dict[str, object]:
        async with self._state_lock:
            state = _normalize_state(self.conversation_states.get(convo_id), convo_id)
            slots = dict(state.get("slots") or {})
            slots[endpoint] = int(slot_id)
            state["slots"] = slots
            state["updated_at"] = _utc_now()
            self.conversation_states[convo_id] = deepcopy(state)
            return _normalize_state(state, convo_id)

    @staticmethod
    def _apply_state_update(
        state: Dict[str, object],
        *,
        route_endpoint: Optional[str],
        reasoning_effort: Optional[str],
        valid_route_endpoints: Optional[List[str]],
        clear_route: bool,
        allow_route_switch: bool,
        allow_reasoning_switch: bool,
    ) -> Dict[str, object]:
        updated = _normalize_state(state, str(state.get("convo_id") or ""))
        current_route = str(updated.get("route_endpoint") or "").strip() or None
        current_reasoning = str(updated.get("reasoning_effort") or "").strip() or None
        valid_routes = (
            set(valid_route_endpoints or [])
            if valid_route_endpoints is not None
            else None
        )
        route_stale = bool(
            current_route
            and valid_routes is not None
            and current_route not in valid_routes
        )
        comparable_route = None if route_stale else current_route
        conflicts: Dict[str, str] = {}
        switched: Dict[str, str] = {}

        if route_endpoint and comparable_route and comparable_route != route_endpoint:
            if allow_route_switch:
                switched["route_endpoint"] = comparable_route
            else:
                conflicts["route_endpoint"] = comparable_route
        if (
            reasoning_effort
            and current_reasoning
            and current_reasoning != reasoning_effort
        ):
            if allow_reasoning_switch:
                switched["reasoning_effort"] = current_reasoning
            else:
                conflicts["reasoning_effort"] = current_reasoning

        if conflicts:
            return {
                "state": updated,
                "conflict": True,
                "conflicts": conflicts,
                "route_stale": route_stale,
            }

        if clear_route or route_stale:
            updated["route_endpoint"] = None
        if route_endpoint:
            updated["route_endpoint"] = route_endpoint
        if reasoning_effort:
            updated["reasoning_effort"] = reasoning_effort

        if (
            clear_route
            or route_stale
            or route_endpoint
            or reasoning_effort
            or updated.get("updated_at") is None
        ):
            updated["updated_at"] = _utc_now()

        return {
            "state": updated,
            "conflict": False,
            "conflicts": {},
            "route_stale": route_stale,
            "switched": switched,
        }

    def clear(self) -> None:
        self.conversations.clear()
        self.updated_at.clear()
        self.conversation_states.clear()


class SQLiteHistoryStore(HistoryStore):
    backend_name = "sqlite"

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self._conn: Optional[aiosqlite.Connection] = None
        self._write_lock = asyncio.Lock()

    async def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self.db_path))
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                convo_id TEXT NOT NULL,
                message_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """)
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversation_messages_convo_id_id
            ON conversation_messages (convo_id, id)
            """)
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_state (
                convo_id TEXT PRIMARY KEY,
                route_endpoint TEXT,
                reasoning_effort TEXT,
                slots_json TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL
            )
            """)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def get_conversation(self, convo_id: str) -> Optional[List[Dict]]:
        conn = self._require_connection()
        cursor = await conn.execute(
            """
            SELECT message_json
            FROM conversation_messages
            WHERE convo_id = ?
            ORDER BY id ASC
            """,
            (convo_id,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        if not rows:
            return None
        return [json.loads(row[0]) for row in rows]

    async def append_messages(self, convo_id: str, messages: List[Dict]) -> None:
        if not messages:
            return

        conn = self._require_connection()
        created_at = _utc_now()
        payloads = [
            (convo_id, json.dumps(message), created_at)
            for message in deepcopy(messages)
        ]
        async with self._write_lock:
            await self._begin_immediate(conn)
            try:
                await conn.executemany(
                    """
                    INSERT INTO conversation_messages (convo_id, message_json, created_at)
                    VALUES (?, ?, ?)
                    """,
                    payloads,
                )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    async def list_conversations(self) -> List[Dict[str, object]]:
        conn = self._require_connection()
        cursor = await conn.execute("""
            SELECT convo_id, MAX(created_at) AS last_updated, COUNT(*) AS message_count
            FROM conversation_messages
            GROUP BY convo_id
            ORDER BY last_updated DESC, convo_id ASC
            """)
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            {
                "convo_id": row[0],
                "last_updated": row[1],
                "message_count": row[2],
            }
            for row in rows
        ]

    async def get_conversation_state(self, convo_id: str) -> Dict[str, object]:
        conn = self._require_connection()
        cursor = await conn.execute(
            """
            SELECT convo_id, route_endpoint, reasoning_effort, slots_json, updated_at
            FROM conversation_state
            WHERE convo_id = ?
            """,
            (convo_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return self._row_to_state(row, convo_id)

    async def update_conversation_state(
        self,
        convo_id: str,
        *,
        route_endpoint: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        valid_route_endpoints: Optional[List[str]] = None,
        clear_route: bool = False,
        allow_route_switch: bool = False,
        allow_reasoning_switch: bool = False,
    ) -> Dict[str, object]:
        conn = self._require_connection()
        async with self._write_lock:
            await self._begin_immediate(conn)
            try:
                cursor = await conn.execute(
                    """
                    SELECT convo_id, route_endpoint, reasoning_effort, slots_json, updated_at
                    FROM conversation_state
                    WHERE convo_id = ?
                    """,
                    (convo_id,),
                )
                row = await cursor.fetchone()
                await cursor.close()
                state = self._row_to_state(row, convo_id)
                result = MemoryHistoryStore._apply_state_update(
                    state,
                    route_endpoint=route_endpoint,
                    reasoning_effort=reasoning_effort,
                    valid_route_endpoints=valid_route_endpoints,
                    clear_route=clear_route,
                    allow_route_switch=allow_route_switch,
                    allow_reasoning_switch=allow_reasoning_switch,
                )
                if result.get("conflict"):
                    await conn.rollback()
                    return result

                await self._write_state(conn, result["state"])
                await conn.commit()
                return result
            except BaseException:
                await conn.rollback()
                raise

    async def set_conversation_slot(
        self, convo_id: str, endpoint: str, slot_id: int
    ) -> Dict[str, object]:
        conn = self._require_connection()
        async with self._write_lock:
            await self._begin_immediate(conn)
            try:
                cursor = await conn.execute(
                    """
                    SELECT convo_id, route_endpoint, reasoning_effort, slots_json, updated_at
                    FROM conversation_state
                    WHERE convo_id = ?
                    """,
                    (convo_id,),
                )
                row = await cursor.fetchone()
                await cursor.close()
                state = self._row_to_state(row, convo_id)
                slots = dict(state.get("slots") or {})
                slots[endpoint] = int(slot_id)
                state["slots"] = slots
                state["updated_at"] = _utc_now()
                await self._write_state(conn, state)
                await conn.commit()
                return state
            except BaseException:
                await conn.rollback()
                raise

    async def _begin_immediate(self, conn: aiosqlite.Connection) -> None:
        if conn.in_transaction:
            await conn.rollback()
        await conn.execute("BEGIN IMMEDIATE")

    async def _write_state(
        self, conn: aiosqlite.Connection, state: Dict[str, object]
    ) -> None:
        await conn.execute(
            """
            INSERT INTO conversation_state (
                convo_id, route_endpoint, reasoning_effort, slots_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(convo_id) DO UPDATE SET
                route_endpoint = excluded.route_endpoint,
                reasoning_effort = excluded.reasoning_effort,
                slots_json = excluded.slots_json,
                updated_at = excluded.updated_at
            """,
            (
                state["convo_id"],
                state.get("route_endpoint"),
                state.get("reasoning_effort"),
                json.dumps(state.get("slots") or {}),
                state.get("updated_at") or _utc_now(),
            ),
        )

    @staticmethod
    def _row_to_state(row, convo_id: str) -> Dict[str, object]:
        if not row:
            return _empty_state(convo_id)
        try:
            slots = json.loads(row[3] or "{}")
        except json.JSONDecodeError:
            slots = {}
        return _normalize_state(
            {
                "convo_id": row[0],
                "route_endpoint": row[1],
                "reasoning_effort": row[2],
                "slots": slots,
                "updated_at": row[4],
            },
            convo_id,
        )

    def _require_connection(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("SQLiteHistoryStore is not initialized")
        return self._conn


def build_history_store_from_env() -> HistoryStore:
    raw_db_path = os.getenv("HISTORY_DB_PATH")
    db_path = Path(raw_db_path.strip()) if raw_db_path and raw_db_path.strip() else DEFAULT_HISTORY_DB_PATH
    return SQLiteHistoryStore(db_path)
