"""SearXNG provider."""

from __future__ import annotations

from .base import HtmlSearchProvider, HtmlSearchSelectors


class SearxngHtmlProvider(HtmlSearchProvider):
    def __init__(self, endpoint: str):
        super().__init__(
            "searxng_html",
            "SearXNG",
            endpoint.rstrip("/"),
            "q",
            HtmlSearchSelectors(
                result=".result",
                title="a",
                url="a",
                snippet=".content",
            ),
            fallback_only=True,
        )

    def apply_common_params(self, params: dict[str, str], args) -> None:
        params["format"] = "html"
        if args.language:
            params["language"] = args.language
