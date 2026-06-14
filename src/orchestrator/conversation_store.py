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

DEFAULT_CONVERSATION_DB_PATH = Path("var/conversations.sqlite3")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_state(conversation_id: str) -> Dict[str, object]:
    return {
        "conversation_id": conversation_id,
        "route_endpoint": None,
        "reasoning_effort": None,
        "slots": {},
        "updated_at": None,
    }


def _normalize_state(row: Optional[Dict[str, object]], conversation_id: str) -> Dict[str, object]:
    state = _empty_state(conversation_id)
    if not row:
        return state

    state.update(row)
    slots = state.get("slots")
    state["slots"] = deepcopy(slots) if isinstance(slots, dict) else {}
    return state


class ConversationStore(ABC):
    backend_name = "unknown"

    @abstractmethod
    async def initialize(self) -> None:
        """Prepare the backing store for use."""

    @abstractmethod
    async def close(self) -> None:
        """Release any backing-store resources."""

    @abstractmethod
    async def get_conversation(self, conversation_id: str) -> Optional[List[Dict]]:
        """Return the stored conversation or None when it does not exist."""

    @abstractmethod
    async def get_conversation_message_records(
        self,
        conversation_id: str,
        *,
        after_id: Optional[int] = None,
        limit: Optional[int] = None,
        newest_first: bool = False,
    ) -> List[Dict]:
        """Return stored messages with durable row IDs and timestamps."""

    @abstractmethod
    async def get_latest_conversation_message_id(
        self, conversation_id: str
    ) -> Optional[int]:
        """Return the latest durable row ID for a conversation."""

    @abstractmethod
    async def append_messages(self, conversation_id: str, messages: List[Dict]) -> None:
        """Append messages to a stored conversation."""

    @abstractmethod
    async def get_thread_state(
        self, conversation_id: str
    ) -> Optional[Dict]:
        """Return the derived thread state for a conversation, if present."""

    @abstractmethod
    async def upsert_thread_state(
        self,
        conversation_id: str,
        *,
        covered_message_id: int,
        state_text: str,
        state_json: Optional[Dict] = None,
        metadata_json: Optional[Dict] = None,
    ) -> Dict:
        """Insert or update the derived thread state for a conversation."""

    @abstractmethod
    async def list_conversations(self) -> List[Dict[str, object]]:
        """Return conversation metadata sorted by most recently updated first."""

    @abstractmethod
    async def get_conversation_control_state(self, conversation_id: str) -> Dict[str, object]:
        """Return route/reasoning/slot metadata for a conversation."""

    @abstractmethod
    async def update_conversation_control_state(
        self,
        conversation_id: str,
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
    async def set_conversation_control_slot(
        self, conversation_id: str, endpoint: str, slot_id: int
    ) -> Dict[str, object]:
        """Persist a best-effort upstream slot mapping."""


class MemoryConversationStore(ConversationStore):
    backend_name = "memory"

    def __init__(self):
        self.conversations: Dict[str, List[Dict]] = {}
        self.conversation_records: Dict[str, List[Dict]] = {}
        self.updated_at: Dict[str, str] = {}
        self.conversation_control_states: Dict[str, Dict[str, object]] = {}
        self.thread_states: Dict[str, Dict] = {}
        self._next_message_id = 1
        self._state_lock = asyncio.Lock()

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def get_conversation(self, conversation_id: str) -> Optional[List[Dict]]:
        messages = self.conversations.get(conversation_id)
        return deepcopy(messages) if messages is not None else None

    async def get_conversation_message_records(
        self,
        conversation_id: str,
        *,
        after_id: Optional[int] = None,
        limit: Optional[int] = None,
        newest_first: bool = False,
    ) -> List[Dict]:
        records = self._records_for_conversation(conversation_id)
        if after_id is not None:
            records = [record for record in records if int(record["id"]) > after_id]
        records = sorted(
            records,
            key=lambda record: int(record["id"]),
            reverse=bool(newest_first),
        )
        if limit is not None:
            records = records[: max(0, int(limit))]
        return deepcopy(records)

    async def get_latest_conversation_message_id(
        self, conversation_id: str
    ) -> Optional[int]:
        records = self._records_for_conversation(conversation_id)
        if not records:
            return None
        return max(int(record["id"]) for record in records)

    async def append_messages(self, conversation_id: str, messages: List[Dict]) -> None:
        if not messages:
            return
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        if conversation_id not in self.conversation_records:
            self.conversation_records[conversation_id] = []
        created_at = _utc_now()
        copied_messages = deepcopy(messages)
        self.conversations[conversation_id].extend(copied_messages)
        for message in copied_messages:
            self.conversation_records[conversation_id].append(
                {
                    "id": self._next_message_id,
                    "conversation_id": conversation_id,
                    "message": deepcopy(message),
                    "created_at": created_at,
                }
            )
            self._next_message_id += 1
        self.updated_at[conversation_id] = created_at

    async def get_thread_state(
        self, conversation_id: str
    ) -> Optional[Dict]:
        state = self.thread_states.get(conversation_id)
        return deepcopy(state) if state is not None else None

    async def upsert_thread_state(
        self,
        conversation_id: str,
        *,
        covered_message_id: int,
        state_text: str,
        state_json: Optional[Dict] = None,
        metadata_json: Optional[Dict] = None,
    ) -> Dict:
        now = _utc_now()
        existing = self.thread_states.get(conversation_id) or {}
        state = {
            "conversation_id": conversation_id,
            "covered_message_id": int(covered_message_id),
            "state_text": str(state_text or ""),
            "state_json": deepcopy(state_json) if isinstance(state_json, dict) else None,
            "metadata_json": deepcopy(metadata_json)
            if isinstance(metadata_json, dict)
            else {},
            "created_at": existing.get("created_at") or now,
            "updated_at": now,
        }
        self.thread_states[conversation_id] = deepcopy(state)
        return deepcopy(state)

    async def list_conversations(self) -> List[Dict[str, object]]:
        conversations = []
        for conversation_id, messages in self.conversations.items():
            last_updated = self.updated_at.get(conversation_id)
            if not last_updated:
                continue
            conversations.append(
                {
                    "conversation_id": conversation_id,
                    "last_updated": last_updated,
                    "message_count": len(messages),
                }
            )

        return sorted(
            conversations,
            key=lambda conversation: str(conversation["last_updated"]),
            reverse=True,
        )

    async def get_conversation_control_state(self, conversation_id: str) -> Dict[str, object]:
        async with self._state_lock:
            return _normalize_state(self.conversation_control_states.get(conversation_id), conversation_id)

    async def update_conversation_control_state(
        self,
        conversation_id: str,
        *,
        route_endpoint: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        valid_route_endpoints: Optional[List[str]] = None,
        clear_route: bool = False,
        allow_route_switch: bool = False,
        allow_reasoning_switch: bool = False,
    ) -> Dict[str, object]:
        async with self._state_lock:
            state = _normalize_state(self.conversation_control_states.get(conversation_id), conversation_id)
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
                self.conversation_control_states[conversation_id] = deepcopy(result["state"])
            return result

    async def set_conversation_control_slot(
        self, conversation_id: str, endpoint: str, slot_id: int
    ) -> Dict[str, object]:
        async with self._state_lock:
            state = _normalize_state(self.conversation_control_states.get(conversation_id), conversation_id)
            slots = dict(state.get("slots") or {})
            slots[endpoint] = int(slot_id)
            state["slots"] = slots
            state["updated_at"] = _utc_now()
            self.conversation_control_states[conversation_id] = deepcopy(state)
            return _normalize_state(state, conversation_id)

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
        updated = _normalize_state(state, str(state.get("conversation_id") or ""))
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
        self.conversation_records.clear()
        self.updated_at.clear()
        self.conversation_control_states.clear()
        self.thread_states.clear()
        self._next_message_id = 1

    def _records_for_conversation(self, conversation_id: str) -> List[Dict]:
        records = self.conversation_records.get(conversation_id)
        if records is not None:
            return deepcopy(records)
        messages = self.conversations.get(conversation_id) or []
        timestamp = self.updated_at.get(conversation_id) or _utc_now()
        return [
            {
                "id": index,
                "conversation_id": conversation_id,
                "message": deepcopy(message),
                "created_at": timestamp,
            }
            for index, message in enumerate(messages, start=1)
        ]


class SQLiteConversationStore(ConversationStore):
    backend_name = "sqlite"

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self._conn: Optional[aiosqlite.Connection] = None
        self._write_lock = asyncio.Lock()

    async def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self.db_path))
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._drop_legacy_schema_if_needed()
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                message_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """)
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversation_messages_conversation_id_id
            ON conversation_messages (conversation_id, id)
            """)
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_control_state (
                conversation_id TEXT PRIMARY KEY,
                route_endpoint TEXT,
                reasoning_effort TEXT,
                slots_json TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL
            )
            """)
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS thread_state (
                conversation_id TEXT PRIMARY KEY,
                covered_message_id INTEGER NOT NULL DEFAULT 0,
                state_text TEXT NOT NULL DEFAULT '',
                state_json TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """)
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_thread_state_updated
            ON thread_state (updated_at DESC)
            """)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def get_conversation(self, conversation_id: str) -> Optional[List[Dict]]:
        conn = self._require_connection()
        cursor = await conn.execute(
            """
            SELECT message_json
            FROM conversation_messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        if not rows:
            return None
        return [json.loads(row[0]) for row in rows]

    async def get_conversation_message_records(
        self,
        conversation_id: str,
        *,
        after_id: Optional[int] = None,
        limit: Optional[int] = None,
        newest_first: bool = False,
    ) -> List[Dict]:
        conn = self._require_connection()
        filters = ["conversation_id = ?"]
        params: list[object] = [conversation_id]
        if after_id is not None:
            filters.append("id > ?")
            params.append(int(after_id))
        order = "DESC" if newest_first else "ASC"
        limit_sql = ""
        if limit is not None:
            limit_sql = " LIMIT ?"
            params.append(max(0, int(limit)))
        cursor = await conn.execute(
            f"""
            SELECT id, conversation_id, message_json, created_at
            FROM conversation_messages
            WHERE {' AND '.join(filters)}
            ORDER BY id {order}
            {limit_sql}
            """,
            params,
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            {
                "id": int(row[0]),
                "conversation_id": row[1],
                "message": json.loads(row[2]),
                "created_at": row[3],
            }
            for row in rows
        ]

    async def get_latest_conversation_message_id(
        self, conversation_id: str
    ) -> Optional[int]:
        conn = self._require_connection()
        cursor = await conn.execute(
            """
            SELECT MAX(id)
            FROM conversation_messages
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if not row or row[0] is None:
            return None
        return int(row[0])

    async def append_messages(self, conversation_id: str, messages: List[Dict]) -> None:
        if not messages:
            return

        conn = self._require_connection()
        created_at = _utc_now()
        payloads = [
            (conversation_id, json.dumps(message), created_at)
            for message in deepcopy(messages)
        ]
        async with self._write_lock:
            await self._begin_immediate(conn)
            try:
                await conn.executemany(
                    """
                    INSERT INTO conversation_messages (conversation_id, message_json, created_at)
                    VALUES (?, ?, ?)
                    """,
                    payloads,
                )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    async def get_thread_state(
        self, conversation_id: str
    ) -> Optional[Dict]:
        conn = self._require_connection()
        cursor = await conn.execute(
            """
            SELECT conversation_id, covered_message_id, state_text, state_json,
                   metadata_json, created_at, updated_at
            FROM thread_state
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return self._row_to_thread_state(row)

    async def upsert_thread_state(
        self,
        conversation_id: str,
        *,
        covered_message_id: int,
        state_text: str,
        state_json: Optional[Dict] = None,
        metadata_json: Optional[Dict] = None,
    ) -> Dict:
        conn = self._require_connection()
        now = _utc_now()
        state_json_text = (
            json.dumps(state_json, ensure_ascii=False)
            if isinstance(state_json, dict)
            else None
        )
        metadata_json_text = json.dumps(metadata_json or {}, ensure_ascii=False)
        async with self._write_lock:
            await self._begin_immediate(conn)
            try:
                await conn.execute(
                    """
                    INSERT INTO thread_state (
                        conversation_id, covered_message_id, state_text, state_json,
                        metadata_json, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(conversation_id) DO UPDATE SET
                        covered_message_id = excluded.covered_message_id,
                        state_text = excluded.state_text,
                        state_json = excluded.state_json,
                        metadata_json = excluded.metadata_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        conversation_id,
                        int(covered_message_id),
                        str(state_text or ""),
                        state_json_text,
                        metadata_json_text,
                        now,
                        now,
                    ),
                )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise
        state = await self.get_thread_state(conversation_id)
        if state is None:
            raise RuntimeError(f"failed to persist thread state: {conversation_id}")
        return state

    async def list_conversations(self) -> List[Dict[str, object]]:
        conn = self._require_connection()
        cursor = await conn.execute("""
            SELECT conversation_id, MAX(created_at) AS last_updated, COUNT(*) AS message_count
            FROM conversation_messages
            GROUP BY conversation_id
            ORDER BY last_updated DESC, conversation_id ASC
            """)
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            {
                "conversation_id": row[0],
                "last_updated": row[1],
                "message_count": row[2],
            }
            for row in rows
        ]

    async def get_conversation_control_state(self, conversation_id: str) -> Dict[str, object]:
        conn = self._require_connection()
        cursor = await conn.execute(
            """
            SELECT conversation_id, route_endpoint, reasoning_effort, slots_json, updated_at
            FROM conversation_control_state
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return self._row_to_state(row, conversation_id)

    async def update_conversation_control_state(
        self,
        conversation_id: str,
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
                    SELECT conversation_id, route_endpoint, reasoning_effort, slots_json, updated_at
                    FROM conversation_control_state
                    WHERE conversation_id = ?
                    """,
                    (conversation_id,),
                )
                row = await cursor.fetchone()
                await cursor.close()
                state = self._row_to_state(row, conversation_id)
                result = MemoryConversationStore._apply_state_update(
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

    async def set_conversation_control_slot(
        self, conversation_id: str, endpoint: str, slot_id: int
    ) -> Dict[str, object]:
        conn = self._require_connection()
        async with self._write_lock:
            await self._begin_immediate(conn)
            try:
                cursor = await conn.execute(
                    """
                    SELECT conversation_id, route_endpoint, reasoning_effort, slots_json, updated_at
                    FROM conversation_control_state
                    WHERE conversation_id = ?
                    """,
                    (conversation_id,),
                )
                row = await cursor.fetchone()
                await cursor.close()
                state = self._row_to_state(row, conversation_id)
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

    async def _drop_legacy_schema_if_needed(self) -> None:
        conn = self._require_connection()
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
        tables = {str(row[0]) for row in await cursor.fetchall()}
        await cursor.close()

        legacy_tables = {
            "conversation_state",
            "compacted_conversation_state",
        } & tables
        legacy_columns = False
        if "conversation_messages" in tables:
            cursor = await conn.execute("PRAGMA table_info(conversation_messages)")
            columns = {str(row[1]) for row in await cursor.fetchall()}
            await cursor.close()
            legacy_columns = "convo_id" in columns

        if not legacy_tables and not legacy_columns:
            return

        logger.warning(
            "Legacy conversation schema detected at %s; dropping conversation tables",
            self.db_path,
        )
        for table in (
            "thread_state",
            "compacted_conversation_state",
            "conversation_control_state",
            "conversation_state",
            "conversation_messages",
        ):
            await conn.execute(f"DROP TABLE IF EXISTS {table}")
        await conn.commit()

    async def _write_state(
        self, conn: aiosqlite.Connection, state: Dict[str, object]
    ) -> None:
        await conn.execute(
            """
            INSERT INTO conversation_control_state (
                conversation_id, route_endpoint, reasoning_effort, slots_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                route_endpoint = excluded.route_endpoint,
                reasoning_effort = excluded.reasoning_effort,
                slots_json = excluded.slots_json,
                updated_at = excluded.updated_at
            """,
            (
                state["conversation_id"],
                state.get("route_endpoint"),
                state.get("reasoning_effort"),
                json.dumps(state.get("slots") or {}),
                state.get("updated_at") or _utc_now(),
            ),
        )

    @staticmethod
    def _row_to_state(row, conversation_id: str) -> Dict[str, object]:
        if not row:
            return _empty_state(conversation_id)
        try:
            slots = json.loads(row[3] or "{}")
        except json.JSONDecodeError:
            slots = {}
        return _normalize_state(
            {
                "conversation_id": row[0],
                "route_endpoint": row[1],
                "reasoning_effort": row[2],
                "slots": slots,
                "updated_at": row[4],
            },
            conversation_id,
        )

    @staticmethod
    def _row_to_thread_state(row) -> Optional[Dict]:
        if not row:
            return None
        try:
            state_json = json.loads(row[3]) if row[3] else None
        except json.JSONDecodeError:
            state_json = None
        try:
            metadata_json = json.loads(row[4] or "{}")
        except json.JSONDecodeError:
            metadata_json = {}
        return {
            "conversation_id": row[0],
            "covered_message_id": int(row[1]),
            "state_text": row[2] or "",
            "state_json": state_json if isinstance(state_json, dict) else None,
            "metadata_json": metadata_json if isinstance(metadata_json, dict) else {},
            "created_at": row[5],
            "updated_at": row[6],
        }

    def _require_connection(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("SQLiteConversationStore is not initialized")
        return self._conn


def build_conversation_store_from_env() -> ConversationStore:
    raw_db_path = os.getenv("CONVERSATION_DB_PATH")
    db_path = (
        Path(raw_db_path.strip())
        if raw_db_path and raw_db_path.strip()
        else DEFAULT_CONVERSATION_DB_PATH
    )
    return SQLiteConversationStore(db_path)
