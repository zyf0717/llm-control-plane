import logging
import uuid

from shiny import App

from src.logging_config import configure_logging

configure_logging()

from .app_server import server
from .app_ui import app_ui

logger = logging.getLogger(__name__)


class WebSocketDiagnostics:
    """Record which side closes a dashboard connection without logging chat data."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "websocket":
            return await self.app(scope, receive, send)

        connection_id = uuid.uuid4().hex[:12]

        async def receive_logged():
            message = await receive()
            if message["type"] == "websocket.disconnect":
                logger.info(
                    "Dashboard WebSocket disconnected: connection=%s code=%s reason=%r",
                    connection_id, message.get("code"), message.get("reason", ""),
                )
            return message

        async def send_logged(message):
            if message["type"] in {"websocket.accept", "websocket.close"}:
                logger.info(
                    "Dashboard WebSocket %s: connection=%s code=%s reason=%r",
                    message["type"].split(".")[1], connection_id,
                    message.get("code"), message.get("reason", ""),
                )
            await send(message)

        try:
            await self.app(scope, receive_logged, send_logged)
        except Exception:
            logger.exception("Dashboard WebSocket failed: connection=%s", connection_id)
            raise


app = App(app_ui, server)
app.starlette_app.add_middleware(WebSocketDiagnostics)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=12341)
