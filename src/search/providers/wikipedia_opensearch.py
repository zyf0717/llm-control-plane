"""Wikipedia OpenSearch provider."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from urllib.parse import urlencode

from ..normalize import dedupe_results, normalize_snippet
from ..types import SearchArgs, SearchRequest, SearchResult
from .base import BaseSearchProvider


class WikipediaOpenSearchProvider(BaseSearchProvider):
    id = "wikipedia_opensearch"
    engine = "Wikipedia"
    response_type = "json"
    fallback_only = True

    def build_request(self, args: SearchArgs) -> SearchRequest:
        params = {
            "action": "opensearch",
            "search": args.query,
            "limit": str(args.count or 10),
            "namespace": "0",
            "format": "json",
        }
        return SearchRequest(
            url=(
                "https://en.wikipedia.org/w/api.php"
                f"?{urlencode(params)}"
            ),
            headers={"Accept": "application/json"},
        )

    def parse(self, raw: str, args: SearchArgs) -> list[SearchResult]:
        payload = json.loads(raw)
        if not isinstance(payload, list) or len(payload) < 4:
            return []

        titles = payload[1] if isinstance(payload[1], list) else []
        snippets = payload[2] if isinstance(payload[2], list) else []
        urls = payload[3] if isinstance(payload[3], list) else []

        results: list[SearchResult] = []
        for title, snippet, url in zip(titles, snippets, urls, strict=False):
            if not title or not url:
                continue
            results.append(
                SearchResult(
                    title=str(title).strip(),
                    url=str(url).strip(),
                    snippet=normalize_snippet(str(snippet)),
                    rank=len(results) + 1,
                    provider=self.id,
                    engine=self.engine,
                    fetched_at=datetime.now(UTC).isoformat(),
                )
            )

        return dedupe_results(results)
