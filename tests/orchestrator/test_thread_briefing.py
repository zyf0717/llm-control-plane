import pytest

from src.orchestrator.thread_briefing import (
    build_bounded_chat_messages,
    build_thread_briefing_bundle,
    enrich_workflow_params_with_thread_bundle,
    format_message_records,
)
from src.orchestrator.conversation_store import MemoryConversationStore


@pytest.mark.asyncio
async def test_thread_bundle_includes_thread_state_and_recent_messages():
    store = MemoryConversationStore()
    await store.append_messages(
        "session-1",
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
        ],
    )
    records = await store.get_conversation_message_records("session-1")
    await store.upsert_thread_state(
        "session-1",
        covered_message_id=records[1]["id"],
        state_text="prior summary",
    )

    bundle = await build_thread_briefing_bundle(
        store,
        source_conversation_id="session-1",
        recent_message_count=2,
    )

    assert bundle["thread_state_text"] == "prior summary"
    assert "prior summary" in bundle["thread_briefing"]
    assert "two" in bundle["recent_conversation_messages"]
    assert "three" in bundle["recent_conversation_messages"]
    assert "one" not in bundle["recent_conversation_messages"]
    assert bundle["thread_metadata"]["covered_message_id"] == records[1]["id"]
    assert bundle["thread_metadata"]["latest_message_id"] == records[-1]["id"]


def test_format_message_records_truncates_oldest_side_first():
    records = [
        {
            "id": 1,
            "conversation_id": "session-1",
            "created_at": "2026-01-01T00:00:00+00:00",
            "message": {"role": "user", "content": "old " * 40},
        },
        {
            "id": 2,
            "conversation_id": "session-1",
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
        bundle={"thread_briefing": "summary and tail"},
    )

    assert messages[0] == {"role": "system", "content": "Base system"}
    assert messages[1]["role"] == "system"
    assert "summary and tail" in messages[1]["content"]
    assert messages[-1] == {"role": "user", "content": "current"}
    assert {"role": "user", "content": "old"} not in messages


def test_workflow_param_enrichment_preserves_existing_briefing_and_prompt():
    bundle = {
        "thread_state_text": "summary",
        "recent_conversation_messages": "tail",
        "thread_briefing": "server briefing",
        "thread_metadata": {"source_conversation_id": "session-1"},
    }
    params = {
        "latest_user_prompt": "do this",
        "thread_briefing": "manual briefing",
    }

    enriched = enrich_workflow_params_with_thread_bundle(params, bundle)

    assert params["thread_briefing"] == "manual briefing"
    assert enriched["latest_user_prompt"] == "do this"
    assert enriched["thread_state_text"] == "summary"
    assert enriched["recent_conversation_messages"] == "tail"
    assert enriched["thread_briefing"].startswith("manual briefing")
    assert "server briefing" in enriched["thread_briefing"]


def test_workflow_param_enrichment_sets_blank_briefing():
    enriched = enrich_workflow_params_with_thread_bundle(
        {},
        {
            "thread_briefing": "server briefing",
            "thread_metadata": {},
        },
    )

    assert enriched["thread_briefing"] == "server briefing"
