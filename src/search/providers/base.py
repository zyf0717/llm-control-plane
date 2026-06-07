"""Reusable provider base classes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Optional
from urllib.parse import urlencode

from bs4 import BeautifulSoup

from ..normalize import clean_search_url, dedupe_results, normalize_snippet
from ..types import SearchArgs, SearchRequest, SearchResponse, SearchResult


@dataclass(slots=True)
class HtmlSearchSelectors:
    result: str
    title: str
    url: str
    snippet: Optional[str] = None


@dataclass(slots=True)
class HtmlProviderOptions:
    extra_params: dict[str, str] = field(default_factory=dict)
    challenge_markers: tuple[str, ...] = ()


class BaseSearchProvider:
    """Common search execution flow for providers."""

    id: str
    engine: str
    response_type: str
    fallback_only: bool

    async def search(self, args, client, cache, config, provider_config) -> SearchResponse:
        cache_key = self.cache_key(args)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        request = self.build_request(args)
        raw = await client.fetch_text(request)
        results = self.parse(raw, args)
        results = dedupe_results(results, config.effective_count(args.count))

        response = SearchResponse(
            query=args.query,
            provider=self.id,
            results=results,
        )
        cache.set(cache_key, response, config.effective_ttl(args, provider_config))
        return response

    def cache_key(self, args: SearchArgs) -> str:
        raw = json.dumps(
            {
                "query": args.query,
                "provider": self.id,
                "language": args.language,
                "region": args.region,
                "safe_search": args.safe_search,
                "freshness": args.freshness,
                "count": args.count,
            },
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class HtmlSearchProvider(BaseSearchProvider):
    """Selector-driven HTML search provider."""

    response_type = "html"

    def __init__(
        self,
        provider_id: str,
        engine: str,
        endpoint: str,
        query_param: str,
        selectors: HtmlSearchSelectors,
        *,
        fallback_only: bool = False,
        options: Optional[HtmlProviderOptions] = None,
    ):
        self.id = provider_id
        self.engine = engine
        self.endpoint = endpoint
        self.query_param = query_param
        self.selectors = selectors
        self.fallback_only = fallback_only
        self.options = options or HtmlProviderOptions()

    def build_request(self, args: SearchArgs) -> SearchRequest:
        params = dict(self.options.extra_params)
        params[self.query_param] = args.query
        self.apply_common_params(params, args)
        return SearchRequest(url=f"{self.endpoint}?{urlencode(params)}")

    def apply_common_params(self, params: dict[str, str], args: SearchArgs) -> None:
        """Provider-specific params can be injected by subclasses."""

    def parse(self, raw: str, args: SearchArgs) -> list[SearchResult]:
        self._raise_if_challenge(raw)

        soup = BeautifulSoup(raw, "html.parser")
        results: list[SearchResult] = []

        for node in soup.select(self.selectors.result):
            title_node = node.select_one(self.selectors.title)
            if title_node is None:
                continue

            url_node = node.select_one(self.selectors.url) if self.selectors.url else None
            raw_url = title_node.get("href") or (url_node.get("href") if url_node else None)
            title = " ".join(title_node.get_text(" ", strip=True).split())
            snippet = None
            if self.selectors.snippet:
                snippet_node = node.select_one(self.selectors.snippet)
                if snippet_node is not None:
                    snippet = normalize_snippet(snippet_node.get_text(" ", strip=True))

            cleaned_url = clean_search_url(raw_url or "")
            if not title or not cleaned_url:
                continue

            results.append(
                SearchResult(
                    title=title,
                    url=cleaned_url,
                    snippet=snippet,
                    rank=len(results) + 1,
                    provider=self.id,
                    engine=self.engine,
                    fetched_at=datetime.now(UTC).isoformat(),
                )
            )

        return dedupe_results(results)

    def _raise_if_challenge(self, raw: str) -> None:
        haystack = raw.lower()
        for marker in self.options.challenge_markers:
            if marker.lower() in haystack:
                raise ValueError(f"{self.id}: challenge page detected")
