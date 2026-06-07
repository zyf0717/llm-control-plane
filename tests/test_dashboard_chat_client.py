import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.dashboard import chat_client
from src.dashboard.chat_client import build_chat_messages, stream_chat_response


class _FakeStreamingResponse:
    def __init__(self, *, headers, lines):
        self.headers = headers
        self._lines = lines
        self.raise_for_status = Mock()

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStreamContext:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


def test_build_chat_messages_with_user_only():
    messages = build_chat_messages(text="Hello")

    assert messages == [{"role": "user", "content": "Hello"}]


def test_build_chat_messages_includes_turn_local_search_context_after_system_prompt():
    messages = build_chat_messages(
        text="Need sources",
        system_prompt="Be concise.",
        extra_turn_messages=[
            {"role": "system", "content": '{"source":"web_search"}'},
            {"role": "assistant", "content": "ignored but allowed"},
            {"role": "", "content": "skip"},
        ],
    )

    assert messages == [
        {"role": "system", "content": "Be concise."},
        {"role": "system", "content": '{"source":"web_search"}'},
        {"role": "assistant", "content": "ignored but allowed"},
        {"role": "user", "content": "Need sources"},
    ]


@pytest.mark.asyncio
async def test_stream_chat_response_does_not_emit_metadata_for_content_only_chunks(
    monkeypatch,
):
    monkeypatch.setattr(chat_client, "PROXY_BASE_URL", "http://proxy.local")
    monkeypatch.setattr(chat_client, "API_KEY_ID", "test-id")
    monkeypatch.setattr(chat_client, "API_KEY_SECRET", "test-secret")

    response = _FakeStreamingResponse(
        headers={
            "x-route-decision": "gpu-node-a",
            "x-rag-endpoint": "http://rag.local/context",
        },
        lines=[
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            (
                "data: "
                + json.dumps(
                    {"choices": [{"delta": {}}], "usage": {"prompt_tokens": 12}}
                )
            ),
            "data: [DONE]",
        ],
    )
    metadata_events = []
    chunks = []

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.stream.return_value = _FakeStreamContext(response)

        async for chunk in stream_chat_response(
            endpoint_key="Auto",
            text="hello",
            endpoints_dict={},
            on_metadata=metadata_events.append,
        ):
            chunks.append(chunk)

    assert "".join(chunks) == "Hello"
    assert metadata_events == [
        {
            "routing": {"decision": "gpu-node-a"},
            "rag": {"endpoint": "http://rag.local/context"},
        },
        {
            "routing": {"decision": "gpu-node-a"},
            "rag": {"endpoint": "http://rag.local/context"},
            "usage": {"prompt_tokens": 12},
        },
    ]
