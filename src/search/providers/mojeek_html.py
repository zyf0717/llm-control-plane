"""Mojeek HTML provider."""

from __future__ import annotations

from .base import HtmlSearchProvider, HtmlSearchSelectors


class MojeekHtmlProvider(HtmlSearchProvider):
    def __init__(self):
        super().__init__(
            "mojeek_html",
            "Mojeek",
            "https://www.mojeek.com/search",
            "q",
            HtmlSearchSelectors(
                result=".result",
                title="a",
                url="a",
                snippet=".s",
            ),
        )
