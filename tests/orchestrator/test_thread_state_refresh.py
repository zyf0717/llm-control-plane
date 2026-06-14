import pytest

from src.orchestrator.thread_state_refresh import (
    ThreadStateRefreshSettings,
    maybe_refresh_thread_state,
)
from src.orchestrator.conversation_store import MemoryConversationStore


class CapturingLLMClient:
    def __init__(self):
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return {"text": kwargs["prompt"], "metadata": {"endpoint": kwargs["endpoint"]}}


@pytest.mark.asyncio
async def test_thread_state_refresh_skips_below_thresholds():
    store = MemoryConversationStore()
    llm = CapturingLLMClient()
    await store.append_messages("session-1", [{"role": "user", "content": "short"}])

    result = await maybe_refresh_thread_state(
        conversation_store=store,
        llm_client=llm,
        source_conversation_id="session-1",
        endpoint="node-a",
        settings=ThreadStateRefreshSettings(
            recent_message_count=0,
            trigger_delta_chars=1000,
            trigger_delta_messages=10,
        ),
    )

    assert result["updated"] is False
    assert result["reason"] == "below-threshold"
    assert llm.calls == []
    assert await store.get_thread_state("session-1") is None


@pytest.mark.asyncio
async def test_thread_state_refresh_force_calls_llm_and_persists_state():
    store = MemoryConversationStore()
    llm = CapturingLLMClient()
    await store.append_messages(
        "session-1",
        [
            {"role": "user", "content": "alpha"},
            {"role": "assistant", "content": "beta"},
        ],
    )

    result = await maybe_refresh_thread_state(
        conversation_store=store,
        llm_client=llm,
        source_conversation_id="session-1",
        endpoint="node-a",
        settings=ThreadStateRefreshSettings(
            compression_chunk_chars=100000,
            compression_target_chars=90000,
            compression_max_output_chars=100000,
        ),
        force=True,
    )

    state = await store.get_thread_state("session-1")
    latest_id = await store.get_latest_conversation_message_id("session-1")
    assert result["updated"] is True
    assert len(llm.calls) == 1
    assert llm.calls[0]["skip_conversation"] is True
    assert state is not None
    assert state["covered_message_id"] == latest_id
    assert "New raw conversation messages" in state["state_text"]


@pytest.mark.asyncio
async def test_thread_state_refresh_includes_previous_state_and_advances():
    store = MemoryConversationStore()
    llm = CapturingLLMClient()
    settings = ThreadStateRefreshSettings(
        recent_message_count=0,
        trigger_delta_chars=1,
        trigger_delta_messages=1,
        compression_chunk_chars=100000,
        compression_target_chars=90000,
        compression_max_output_chars=100000,
    )
    await store.append_messages("session-1", [{"role": "user", "content": "alpha"}])
    first = await maybe_refresh_thread_state(
        conversation_store=store,
        llm_client=llm,
        source_conversation_id="session-1",
        endpoint="node-a",
        settings=settings,
    )
    await store.append_messages("session-1", [{"role": "assistant", "content": "beta"}])

    second = await maybe_refresh_thread_state(
        conversation_store=store,
        llm_client=llm,
        source_conversation_id="session-1",
        endpoint="node-a",
        settings=settings,
    )

    assert first["updated"] is True
    assert second["updated"] is True
    assert "Previous thread state" in llm.calls[1]["prompt"]
    assert "alpha" in llm.calls[1]["prompt"]
    assert "beta" in llm.calls[1]["prompt"]
    assert second["covered_message_id"] > first["covered_message_id"]
