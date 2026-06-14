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


class _FakeStreamSource:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeResponse:
    def __init__(self, *, headers, payload):
        self.headers = headers
        self._payload = payload
        self.raise_for_status = Mock()

    def json(self):
        return self._payload


def test_build_chat_messages_with_user_only():
    messages = build_chat_messages(text="Hello")

    assert messages == [{"role": "user", "content": "Hello"}]


def test_build_chat_messages_includes_turn_local_search_evidence_after_system_prompt():
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
            "x-trace-id": "abc123",
            "x-route-decision": "gpu-node-a",
            "x-retrieval-endpoint": "http://retrieval.local/evidence",
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
        mock_client.stream.return_value = _FakeStreamSource(response)

        async for chunk in stream_chat_response(
            endpoint_key="smart",
            text="hello",
            endpoints_dict={},
            on_metadata=metadata_events.append,
        ):
            chunks.append(chunk)

    assert "".join(chunks) == "Hello"
    assert metadata_events == [
        {
            "trace": {"id": "abc123"},
            "routing": {"decision": "gpu-node-a"},
            "retrieval": {"endpoint": "http://retrieval.local/evidence"},
        },
        {
            "trace": {"id": "abc123"},
            "routing": {"decision": "gpu-node-a"},
            "retrieval": {"endpoint": "http://retrieval.local/evidence"},
            "usage": {"prompt_tokens": 12},
        },
    ]


@pytest.mark.asyncio
async def test_concrete_endpoint_request_opts_into_switches_and_surfaces_warning(
    monkeypatch,
):
    monkeypatch.setattr(chat_client, "PROXY_BASE_URL", "http://proxy.local")
    monkeypatch.setattr(chat_client, "API_KEY_ID", "test-id")
    monkeypatch.setattr(chat_client, "API_KEY_SECRET", "test-secret")

    response = _FakeResponse(
        headers={
            "x-route-switched": "true",
            "x-route-previous": "node-a",
            "x-route-decision": "node-b",
            "x-warning": "Conversation endpoint/reasoning changed",
        },
        payload={"choices": [{"message": {"content": "Answer"}}]},
    )
    metadata_events = []
    chunks = []

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = Mock()
        mock_client.post = AsyncMock(return_value=response)
        mock_client_class.return_value.__aenter__.return_value = mock_client

        async for chunk in stream_chat_response(
            endpoint_key="node-b",
            text="hello",
            endpoints_dict={"node-b": {"url": "http://node-b.local"}},
            stream=False,
            reasoning_effort="high",
            conversation_id="conversation-1",
            on_metadata=metadata_events.append,
        ):
            chunks.append(chunk)

    call = mock_client.post.call_args
    assert call.args[0] == "http://proxy.local/node-b"
    assert call.kwargs["headers"]["X-Allow-Route-Switch"] == "true"
    assert "X-Allow-Reasoning-Switch" not in call.kwargs["headers"]
    assert call.kwargs["headers"]["X-Conversation-ID"] == "conversation-1"
    rendered = "".join(chunks)
    assert rendered.startswith(
        "**Warning:** conversation endpoint changed from node-a to node-b; "
        "full history was sent to the selected endpoint."
    )
    assert rendered.endswith("Answer")
    assert metadata_events[0]["warning"] == {
        "message": "Conversation endpoint/reasoning changed"
    }
    assert metadata_events[0]["routing"]["switched"] == "true"


@pytest.mark.asyncio
async def test_smart_endpoint_uses_smart_without_route_switch_opt_in(monkeypatch):
    monkeypatch.setattr(chat_client, "PROXY_BASE_URL", "http://proxy.local")
    monkeypatch.setattr(chat_client, "API_KEY_ID", "test-id")
    monkeypatch.setattr(chat_client, "API_KEY_SECRET", "test-secret")

    response = _FakeResponse(
        headers={},
        payload={"choices": [{"message": {"content": "Smart answer"}}]},
    )

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = Mock()
        mock_client.post = AsyncMock(return_value=response)
        mock_client_class.return_value.__aenter__.return_value = mock_client

        chunks = [
            chunk
            async for chunk in stream_chat_response(
                endpoint_key="smart",
                text="hello",
                endpoints_dict={},
                stream=False,
                reasoning_effort="none",
            )
        ]

    call = mock_client.post.call_args
    assert call.args[0] == "http://proxy.local/smart"
    assert "X-Allow-Route-Switch" not in call.kwargs["headers"]
    assert "X-Allow-Reasoning-Switch" not in call.kwargs["headers"]
    assert "".join(chunks) == "Smart answer"
