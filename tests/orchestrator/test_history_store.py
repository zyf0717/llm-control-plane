import asyncio
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.orchestrator.history_store import (
    DEFAULT_HISTORY_DB_PATH,
    MemoryHistoryStore,
    SQLiteHistoryStore,
    build_history_store_from_env,
)


class TestHistoryStoreConfiguration:
    def test_build_history_store_uses_default_sqlite_path(self):
        with patch.dict(os.environ, {}, clear=True):
            store = build_history_store_from_env()

        assert isinstance(store, SQLiteHistoryStore)
        assert store.db_path == DEFAULT_HISTORY_DB_PATH

    def test_build_history_store_uses_env_override(self, tmp_path):
        db_path = tmp_path / "custom-history.sqlite3"
        with patch.dict(os.environ, {"HISTORY_DB_PATH": str(db_path)}, clear=True):
            store = build_history_store_from_env()

        assert isinstance(store, SQLiteHistoryStore)
        assert store.db_path == db_path

    def test_build_history_store_uses_default_sqlite_path_when_env_is_blank(self):
        with patch.dict(os.environ, {"HISTORY_DB_PATH": ""}, clear=True):
            store = build_history_store_from_env()

        assert isinstance(store, SQLiteHistoryStore)
        assert store.db_path == DEFAULT_HISTORY_DB_PATH


class TestSQLiteHistoryStore:
    @pytest.mark.asyncio
    async def test_sqlite_history_store_bootstraps_schema(self, tmp_path):
        db_path = tmp_path / "history.sqlite3"
        store = SQLiteHistoryStore(db_path)

        await store.initialize()
        await store.close()

        assert db_path.exists()

    @pytest.mark.asyncio
    async def test_sqlite_history_store_round_trip_messages(self, tmp_path):
        db_path = tmp_path / "history.sqlite3"
        store = SQLiteHistoryStore(db_path)
        await store.initialize()

        try:
            await store.append_messages(
                "session-1",
                [
                    {"role": "system", "content": "System prompt"},
                    {"role": "user", "content": "Hello"},
                ],
            )
            await store.append_messages(
                "session-1", [{"role": "assistant", "content": "Hi"}]
            )

            assert await store.get_conversation("session-1") == [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_sqlite_history_store_lists_conversations_most_recent_first(
        self, tmp_path
    ):
        db_path = tmp_path / "history.sqlite3"
        store = SQLiteHistoryStore(db_path)
        await store.initialize()

        try:
            await store.append_messages(
                "session-1", [{"role": "user", "content": "Hello"}]
            )
            await store.append_messages(
                "session-2",
                [
                    {"role": "user", "content": "Latest"},
                    {"role": "assistant", "content": "Reply"},
                ],
            )

            conversations = await store.list_conversations()

            assert [conversation["convo_id"] for conversation in conversations] == [
                "session-2",
                "session-1",
            ]
            assert [
                conversation["message_count"] for conversation in conversations
            ] == [
                2,
                1,
            ]
            assert all(
                isinstance(conversation["last_updated"], str)
                for conversation in conversations
            )
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_sqlite_history_store_returns_none_for_unknown_conversation(
        self, tmp_path
    ):
        db_path = tmp_path / "history.sqlite3"
        store = SQLiteHistoryStore(db_path)
        await store.initialize()

        try:
            assert await store.get_conversation("missing") is None
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_sqlite_history_store_message_records_and_compacted_state(
        self, tmp_path
    ):
        db_path = tmp_path / "history.sqlite3"
        store = SQLiteHistoryStore(db_path)
        await store.initialize()

        try:
            await store.append_messages(
                "session-1",
                [
                    {"role": "user", "content": "one"},
                    {"role": "assistant", "content": "two"},
                ],
            )
            await store.append_messages(
                "session-1", [{"role": "user", "content": "three"}]
            )

            records = await store.get_conversation_message_records("session-1")
            assert [record["message"]["content"] for record in records] == [
                "one",
                "two",
                "three",
            ]
            assert [record["id"] for record in records] == sorted(
                record["id"] for record in records
            )
            assert await store.get_latest_conversation_message_id("session-1") == records[-1]["id"]

            after_first = await store.get_conversation_message_records(
                "session-1", after_id=records[0]["id"]
            )
            assert [record["message"]["content"] for record in after_first] == [
                "two",
                "three",
            ]
            newest_two = await store.get_conversation_message_records(
                "session-1", limit=2, newest_first=True
            )
            assert [record["message"]["content"] for record in newest_two] == [
                "three",
                "two",
            ]

            state = await store.upsert_compacted_conversation_state(
                "session-1",
                covered_message_id=records[1]["id"],
                state_text="summary",
                state_json={"summary": "summary"},
                metadata_json={"method": "test"},
            )
            assert state["covered_message_id"] == records[1]["id"]
            assert state["state_json"] == {"summary": "summary"}
            assert state["metadata_json"] == {"method": "test"}

            updated = await store.upsert_compacted_conversation_state(
                "session-1",
                covered_message_id=records[-1]["id"],
                state_text="updated",
            )
            assert updated["created_at"] == state["created_at"]
            assert updated["updated_at"] >= state["updated_at"]
            assert (
                await store.get_compacted_conversation_state("session-1")
            )["state_text"] == "updated"
        finally:
            await store.close()


class TestConversationState:
    @pytest.mark.asyncio
    async def test_memory_history_store_message_records_and_compacted_state(self):
        store = MemoryHistoryStore()
        await store.append_messages(
            "session-1",
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
            ],
        )
        records = await store.get_conversation_message_records("session-1")

        assert [record["message"]["content"] for record in records] == ["one", "two"]
        assert await store.get_latest_conversation_message_id("session-1") == records[-1]["id"]

        state = await store.upsert_compacted_conversation_state(
            "session-1",
            covered_message_id=records[-1]["id"],
            state_text="summary",
            metadata_json={"method": "memory"},
        )

        assert state == await store.get_compacted_conversation_state("session-1")

    @pytest.mark.asyncio
    async def test_memory_state_concurrent_first_writes_do_not_create_conflicting_pins(self):
        store = MemoryHistoryStore()

        first, second = await asyncio.gather(
            store.update_conversation_state(
                "session-concurrent",
                route_endpoint="primary",
                reasoning_effort="high",
                valid_route_endpoints=["primary", "secondary"],
            ),
            store.update_conversation_state(
                "session-concurrent",
                route_endpoint="secondary",
                reasoning_effort="high",
                valid_route_endpoints=["primary", "secondary"],
            ),
        )

        final_state = await store.get_conversation_state("session-concurrent")
        assert final_state["route_endpoint"] in {"primary", "secondary"}
        assert final_state["reasoning_effort"] == "high"
        assert sum(1 for result in [first, second] if result["conflict"]) == 1
        assert sum(
            1
            for result in [first, second]
            if result["state"].get("route_endpoint") == final_state["route_endpoint"]
        ) >= 1

    @pytest.mark.asyncio
    async def test_sqlite_state_concurrent_first_writes_do_not_overlap_transactions(
        self, tmp_path
    ):
        db_path = tmp_path / "history.sqlite3"
        store = SQLiteHistoryStore(db_path)
        await store.initialize()

        try:
            first, second = await asyncio.gather(
                store.update_conversation_state(
                    "session-concurrent",
                    route_endpoint="primary",
                    reasoning_effort="medium",
                    valid_route_endpoints=["primary", "secondary"],
                ),
                store.update_conversation_state(
                    "session-concurrent",
                    route_endpoint="secondary",
                    reasoning_effort="medium",
                    valid_route_endpoints=["primary", "secondary"],
                ),
            )

            final_state = await store.get_conversation_state("session-concurrent")
            assert final_state["route_endpoint"] in {"primary", "secondary"}
            assert final_state["reasoning_effort"] == "medium"
            assert sum(1 for result in [first, second] if result["conflict"]) == 1
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_sqlite_write_cancellation_rolls_back_open_transaction(
        self, tmp_path, monkeypatch
    ):
        db_path = tmp_path / "history.sqlite3"
        store = SQLiteHistoryStore(db_path)
        await store.initialize()

        try:
            assert store._conn is not None
            original_commit = store._conn.commit
            calls = 0

            async def flaky_commit():
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise asyncio.CancelledError()
                await original_commit()

            monkeypatch.setattr(store._conn, "commit", flaky_commit)

            with pytest.raises(asyncio.CancelledError):
                await store.append_messages(
                    "session-cancelled",
                    [{"role": "user", "content": "first"}],
                )

            result = await store.update_conversation_state(
                "session-cancelled",
                route_endpoint="primary",
                reasoning_effort="medium",
                valid_route_endpoints=["primary"],
            )

            assert result["conflict"] is False
            assert result["state"]["route_endpoint"] == "primary"
            assert await store.get_conversation("session-cancelled") is None
        finally:
            await store.close()
