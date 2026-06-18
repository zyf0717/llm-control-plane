import pytest

from src.dashboard.app_server import elapsed_seconds_since, stream_with_finalizer
from src.dashboard.prompt_state import (
    RETRIEVAL_CITATION_SUFFIX,
    append_managed_retrieval_suffix,
    build_system_prompt_state,
    extract_first_system_prompt,
    first_turn_system_prompt_to_send,
)


def test_elapsed_seconds_since_uses_non_negative_monotonic_delta():
    assert elapsed_seconds_since(10.0, now=12.25) == 2.25
    assert elapsed_seconds_since(10.0, now=9.5) == 0.0


@pytest.mark.asyncio
async def test_stream_with_finalizer_runs_after_stream_is_consumed():
    finalized = []

    async def source():
        yield "hello"
        assert finalized == []
        yield " world"

    chunks = [
        chunk
        async for chunk in stream_with_finalizer(source(), lambda: finalized.append(True))
    ]

    assert chunks == ["hello", " world"]
    assert finalized == [True]


def test_append_managed_retrieval_suffix_appends_once():
    prompt = append_managed_retrieval_suffix("Be concise.")

    assert prompt == f"Be concise.\n\n{RETRIEVAL_CITATION_SUFFIX}"


def test_append_managed_retrieval_suffix_is_idempotent():
    prompt = append_managed_retrieval_suffix("Be concise.")
    prompt = append_managed_retrieval_suffix(prompt)

    assert prompt.count(RETRIEVAL_CITATION_SUFFIX) == 1


def test_append_managed_retrieval_suffix_uses_suffix_as_prompt_when_empty():
    assert append_managed_retrieval_suffix("") == RETRIEVAL_CITATION_SUFFIX


def test_extract_first_system_prompt_returns_first_system_message():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "system", "content": "System one"},
        {"role": "system", "content": "System two"},
    ]

    assert extract_first_system_prompt(messages) == "System one"


def test_extract_first_system_prompt_returns_none_without_system_message():
    assert extract_first_system_prompt([{"role": "user", "content": "Hello"}]) is None


def test_first_turn_system_prompt_to_send_only_sends_before_start():
    assert first_turn_system_prompt_to_send(" Be concise. ", started=False) == "Be concise."
    assert first_turn_system_prompt_to_send("Be concise.", started=True) is None
    assert first_turn_system_prompt_to_send("", started=False) is None


def test_build_system_prompt_state_tracks_committed_prompt_and_reasoning():
    state = build_system_prompt_state(
        "Be concise.",
        started=True,
        committed_prompt="Original prompt.",
        reasoning_effort="high",
    )

    assert state["prompt"] == "Be concise."
    assert state["committed_prompt"] == "Original prompt."
    assert state["reasoning_effort"] == "high"
    assert state["started"] is True
