"""Core types for lightweight search discovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Protocol


@dataclass(slots=True)
class SearchArgs:
    """Normalized search request arguments."""

    query: str
    count: Optional[int] = None
    provider: str = "auto"
    language: Optional[str] = None
    region: Optional[str] = None
    safe_search: Optional[str] = None
    freshness: Optional[str] = None
    source_text: Optional[str] = None
    use_query_refiner: bool = True
    use_reranker: bool = True
    rerank_source_text: Optional[str] = None


@dataclass(slots=True)
class SearchRequest:
    """HTTP request definition for a provider search."""

    url: str
    method: str = "GET"
    headers: dict[str, str] = field(default_factory=dict)
    body: Optional[str] = None


@dataclass(slots=True)
class SearchResult:
    """Normalized search result."""

    title: str
    url: str
    snippet: Optional[str]
    rank: int
    provider: str
    engine: str
    fetched_at: str
    score: Optional[float] = None
    ranking: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if self.score is None:
            payload.pop("score", None)
        if not self.ranking:
            payload.pop("ranking", None)
        return payload


@dataclass(slots=True)
class SearchResponse:
    """Provider search response."""

    query: str
    provider: str
    results: list[SearchResult]
    degraded: bool = False
    warnings: list[str] = field(default_factory=list)
    original_query: Optional[str] = None
    query_refinement: dict[str, object] = field(default_factory=dict)
    reranking: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["results"] = [result.to_dict() for result in self.results]
        if not self.original_query:
            payload.pop("original_query", None)
        if not self.query_refinement:
            payload.pop("query_refinement", None)
        if not self.reranking:
            payload.pop("reranking", None)
        return payload


class SearchProvider(Protocol):
    """Provider contract."""

    id: str
    engine: str
    response_type: str
    fallback_only: bool

    def build_request(self, args: SearchArgs) -> SearchRequest:
        """Build the search HTTP request."""

    def parse(self, raw: str, args: SearchArgs) -> list[SearchResult]:
        """Parse the raw response into normalized results."""

    async def search(self, args, client, cache, config, provider_config) -> SearchResponse:
        """Execute a provider search."""
