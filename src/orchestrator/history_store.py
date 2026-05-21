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


class MemoryHistoryStore(HistoryStore):
    backend_name = "memory"

    def __init__(self):
        self.conversations: Dict[str, List[Dict]] = {}
        self.updated_at: Dict[str, str] = {}

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
        self.updated_at[convo_id] = datetime.now(timezone.utc).isoformat()

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

    def clear(self) -> None:
        self.conversations.clear()
        self.updated_at.clear()


class SQLiteHistoryStore(HistoryStore):
    backend_name = "sqlite"

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self._conn: Optional[aiosqlite.Connection] = None

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
        created_at = datetime.now(timezone.utc).isoformat()
        payloads = [
            (convo_id, json.dumps(message), created_at)
            for message in deepcopy(messages)
        ]
        await conn.executemany(
            """
            INSERT INTO conversation_messages (convo_id, message_json, created_at)
            VALUES (?, ?, ?)
            """,
            payloads,
        )
        await conn.commit()

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

    def _require_connection(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("SQLiteHistoryStore is not initialized")
        return self._conn


def build_history_store_from_env() -> HistoryStore:
    raw_db_path = os.getenv("HISTORY_DB_PATH")
    if raw_db_path is not None and not raw_db_path.strip():
        logger.info("HISTORY_DB_PATH is blank; using in-memory history store")
        return MemoryHistoryStore()

    db_path = Path(raw_db_path) if raw_db_path else DEFAULT_HISTORY_DB_PATH
    return SQLiteHistoryStore(db_path)
