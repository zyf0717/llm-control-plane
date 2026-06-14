import pytest

from src.orchestrator.conversation_context import (
    build_bounded_chat_messages,
    build_conversation_context_bundle,
    enrich_workflow_params_with_context_bundle,
    format_message_records,
)
from src.orchestrator.history_store import MemoryHistoryStore


@pytest.mark.asyncio
async def test_context_bundle_includes_compacted_state_and_recent_tail():
    store = MemoryHistoryStore()
    await store.append_messages(
        "session-1",
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
        ],
    )
    records = await store.get_conversation_message_records("session-1")
    await store.upsert_compacted_conversation_state(
        "session-1",
        covered_message_id=records[1]["id"],
        state_text="prior summary",
    )

    bundle = await build_conversation_context_bundle(
        store,
        source_convo_id="session-1",
        recent_tail_messages=2,
    )

    assert bundle["compacted_thread_state"] == "prior summary"
    assert "prior summary" in bundle["server_conversation_context"]
    assert "two" in bundle["recent_conversation_tail"]
    assert "three" in bundle["recent_conversation_tail"]
    assert "one" not in bundle["recent_conversation_tail"]
    assert bundle["context_state"]["covered_message_id"] == records[1]["id"]
    assert bundle["context_state"]["latest_message_id"] == records[-1]["id"]


def test_format_message_records_truncates_oldest_side_first():
    records = [
        {
            "id": 1,
            "convo_id": "session-1",
            "created_at": "2026-01-01T00:00:00+00:00",
            "message": {"role": "user", "content": "old " * 40},
        },
        {
            "id": 2,
            "convo_id": "session-1",
            "created_at": "2026-01-01T00:00:01+00:00",
            "message": {"role": "assistant", "content": "new"},
        },
    ]

    text = format_message_records(records, max_chars=120)

    assert "older conversation messages omitted" in text
    assert "[msg:2" in text
    assert "[msg:1" not in text


def test_bounded_chat_messages_preserves_current_user_without_full_history():
    messages = build_bounded_chat_messages(
        stored_messages_before=[
            {"role": "system", "content": "Base system"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old reply"},
        ],
        incoming_messages=[{"role": "user", "content": "current"}],
        bundle={"server_conversation_context": "summary and tail"},
    )

    assert messages[0] == {"role": "system", "content": "Base system"}
    assert messages[1]["role"] == "system"
    assert "summary and tail" in messages[1]["content"]
    assert messages[-1] == {"role": "user", "content": "current"}
    assert {"role": "user", "content": "old"} not in messages


def test_workflow_param_enrichment_preserves_existing_context_and_prompt():
    bundle = {
        "compacted_thread_state": "summary",
        "recent_conversation_tail": "tail",
        "server_conversation_context": "server context",
        "context_state": {"source_convo_id": "session-1"},
    }
    params = {
        "latest_user_prompt": "do this",
        "conversation_context": "manual context",
    }

    enriched = enrich_workflow_params_with_context_bundle(params, bundle)

    assert params["conversation_context"] == "manual context"
    assert enriched["latest_user_prompt"] == "do this"
    assert enriched["compacted_thread_state"] == "summary"
    assert enriched["recent_conversation_tail"] == "tail"
    assert enriched["conversation_context"].startswith("manual context")
    assert "server context" in enriched["conversation_context"]


def test_workflow_param_enrichment_sets_blank_context():
    enriched = enrich_workflow_params_with_context_bundle(
        {},
        {
            "server_conversation_context": "server context",
            "context_state": {},
        },
    )

    assert enriched["conversation_context"] == "server context"
