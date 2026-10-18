import logging

import pytest

from src.dashboard.app import WebSocketDiagnostics


@pytest.mark.asyncio
async def test_websocket_close_codes_are_logged_without_changing_messages(caplog):
    incoming = {"type": "websocket.disconnect", "code": 1006, "reason": "lost connection"}
    outgoing = [
        {"type": "websocket.accept"},
        {"type": "websocket.close", "code": 1011, "reason": "session error"},
    ]
    received = []
    sent = []

    async def receive():
        return incoming

    async def send(message):
        sent.append(message)

    async def endpoint(scope, receive, send):
        for message in outgoing:
            await send(message)
        received.append(await receive())

    with caplog.at_level(logging.INFO, logger="src.dashboard.app"):
        await WebSocketDiagnostics(endpoint)({"type": "websocket"}, receive, send)

    assert received == [incoming]
    assert sent == outgoing
    assert "code=1006" in caplog.text
    assert "code=1011" in caplog.text


@pytest.mark.asyncio
async def test_websocket_error_is_logged_and_propagated(caplog):
    async def endpoint(scope, receive, send):
        raise RuntimeError("session failed")

    with caplog.at_level(logging.ERROR, logger="src.dashboard.app"):
        with pytest.raises(RuntimeError, match="session failed"):
            await WebSocketDiagnostics(endpoint)({"type": "websocket"}, None, None)

    assert "Dashboard WebSocket failed" in caplog.text
    assert caplog.records[-1].exc_info is not None
