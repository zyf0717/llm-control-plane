import asyncio
from unittest.mock import patch

import pytest

from src.dashboard.chat_client import stream_chat_response


class _FakeStreamingResponse:
    def __init__(self):
        self.headers = {"x-route-decision": "worker-a"}
        self.closed = False
        self.release_second_line = asyncio.Event()

    def raise_for_status(self):
        return None

    async def aiter_lines(self):
        yield 'data: {"choices": [{"delta": {"content": "Hello"}}]}'
        await self.release_second_line.wait()
        yield 'data: {"choices": [{"delta": {"content": " world"}}]}'
        yield 'data: [DONE]'

    async def aclose(self):
        self.closed = True
        self.release_second_line.set()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.closed = True
        self.release_second_line.set()


class _FakeAsyncClient:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    def stream(self, *_args, **_kwargs):
        return self.response


@pytest.mark.asyncio
async def test_stream_chat_response_stops_without_error_and_closes_response():
    fake_response = _FakeStreamingResponse()
    stop_event = asyncio.Event()
    states = []

    with patch("httpx.AsyncClient", return_value=_FakeAsyncClient(fake_response)):
        generator = stream_chat_response(
            endpoint_key="worker-a",
            text="Hello",
            endpoints_dict={"worker-a": {}},
            stream=True,
            stop_event=stop_event,
            on_send_button_state=states.append,
        )

        first_chunk = await anext(generator)
        stop_event.set()

        remaining_chunks = []
        async for chunk in generator:
            remaining_chunks.append(chunk)

    assert first_chunk == "Hello"
    assert remaining_chunks == []
    assert fake_response.closed is True
    assert states == ["busy", "ready"]
