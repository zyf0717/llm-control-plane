from shiny import App

from src.logging_config import configure_logging

configure_logging()

from .app_server import server
from .app_ui import app_ui

app = App(app_ui, server)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=12341)
