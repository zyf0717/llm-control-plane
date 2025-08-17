from shiny import App

from .app_server import server
from .app_ui import app_ui

app = App(app_ui, server)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=12341)
