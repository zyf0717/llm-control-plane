"""Marginalia HTML provider."""

from __future__ import annotations

from .base import HtmlSearchProvider, HtmlSearchSelectors


class MarginaliaHtmlProvider(HtmlSearchProvider):
    def __init__(self):
        super().__init__(
            "marginalia_html",
            "Marginalia",
            "https://search.marginalia.nu/search",
            "query",
            HtmlSearchSelectors(
                result=".search-result",
                title="a",
                url="a",
                snippet=".description",
            ),
        )
