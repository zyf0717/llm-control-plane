from shiny import App

from .app_server import server
from .app_ui import app_ui

app = App(app_ui, server)
