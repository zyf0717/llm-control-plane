"""DuckDuckGo HTML provider."""

from __future__ import annotations

from .base import HtmlProviderOptions, HtmlSearchProvider, HtmlSearchSelectors


class DuckDuckGoHtmlProvider(HtmlSearchProvider):
    def __init__(self):
        super().__init__(
            "duckduckgo_html",
            "DuckDuckGo",
            "https://html.duckduckgo.com/html/",
            "q",
            HtmlSearchSelectors(
                result=".result",
                title=".result__a",
                url=".result__a",
                snippet=".result__snippet",
            ),
            options=HtmlProviderOptions(
                challenge_markers=(
                    "bots use duckduckgo too",
                    "challenge-form",
                )
            ),
        )

    def apply_common_params(self, params: dict[str, str], args) -> None:
        if args.region:
            params["kl"] = args.region
        if args.safe_search == "strict":
            params["kp"] = "1"
        elif args.safe_search == "off":
            params["kp"] = "-2"
        if args.freshness:
            params["df"] = {
                "day": "d",
                "week": "w",
                "month": "m",
                "year": "y",
            }.get(args.freshness, "")
            if not params["df"]:
                params.pop("df")
