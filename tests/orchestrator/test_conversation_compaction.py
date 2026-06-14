import pytest

from src.orchestrator.conversation_compaction import (
    ConversationCompactionSettings,
    maybe_refresh_compacted_conversation_state,
)
from src.orchestrator.history_store import MemoryHistoryStore


class CapturingLLMClient:
    def __init__(self):
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return {"text": kwargs["prompt"], "metadata": {"endpoint": kwargs["endpoint"]}}


@pytest.mark.asyncio
async def test_conversation_compaction_skips_below_thresholds():
    store = MemoryHistoryStore()
    llm = CapturingLLMClient()
    await store.append_messages("session-1", [{"role": "user", "content": "short"}])

    result = await maybe_refresh_compacted_conversation_state(
        history_store=store,
        llm_client=llm,
        source_convo_id="session-1",
        endpoint="node-a",
        settings=ConversationCompactionSettings(
            recent_tail_messages=0,
            trigger_delta_chars=1000,
            trigger_delta_messages=10,
        ),
    )

    assert result["updated"] is False
    assert result["reason"] == "below-threshold"
    assert llm.calls == []
    assert await store.get_compacted_conversation_state("session-1") is None


@pytest.mark.asyncio
async def test_conversation_compaction_force_calls_llm_and_persists_state():
    store = MemoryHistoryStore()
    llm = CapturingLLMClient()
    await store.append_messages(
        "session-1",
        [
            {"role": "user", "content": "alpha"},
            {"role": "assistant", "content": "beta"},
        ],
    )

    result = await maybe_refresh_compacted_conversation_state(
        history_store=store,
        llm_client=llm,
        source_convo_id="session-1",
        endpoint="node-a",
        settings=ConversationCompactionSettings(
            compaction_chunk_chars=100000,
            compaction_target_chars=90000,
            compaction_max_output_chars=100000,
        ),
        force=True,
    )

    state = await store.get_compacted_conversation_state("session-1")
    latest_id = await store.get_latest_conversation_message_id("session-1")
    assert result["updated"] is True
    assert len(llm.calls) == 1
    assert llm.calls[0]["skip_history"] is True
    assert state is not None
    assert state["covered_message_id"] == latest_id
    assert "New raw conversation messages" in state["state_text"]


@pytest.mark.asyncio
async def test_conversation_compaction_includes_previous_state_and_advances():
    store = MemoryHistoryStore()
    llm = CapturingLLMClient()
    settings = ConversationCompactionSettings(
        recent_tail_messages=0,
        trigger_delta_chars=1,
        trigger_delta_messages=1,
        compaction_chunk_chars=100000,
        compaction_target_chars=90000,
        compaction_max_output_chars=100000,
    )
    await store.append_messages("session-1", [{"role": "user", "content": "alpha"}])
    first = await maybe_refresh_compacted_conversation_state(
        history_store=store,
        llm_client=llm,
        source_convo_id="session-1",
        endpoint="node-a",
        settings=settings,
    )
    await store.append_messages("session-1", [{"role": "assistant", "content": "beta"}])

    second = await maybe_refresh_compacted_conversation_state(
        history_store=store,
        llm_client=llm,
        source_convo_id="session-1",
        endpoint="node-a",
        settings=settings,
    )

    assert first["updated"] is True
    assert second["updated"] is True
    assert "Previous compacted state" in llm.calls[1]["prompt"]
    assert "alpha" in llm.calls[1]["prompt"]
    assert "beta" in llm.calls[1]["prompt"]
    assert second["covered_message_id"] > first["covered_message_id"]
