"""Lightweight search discovery package."""

from .index import build_search_router
from .query_refiner import (
    SearchQueryRefinement,
    SearchQueryRefiner,
    SearchQueryRefinerConfig,
)
from .reranker import SearchReranker, SearchRerankerConfig, SearchReranking
from .safety import EPHEMERAL_WEB_SEARCH_CONTEXT_MARKER, wrap_search_results
from .search_router import SearchConfig, SearchProviderConfig, SearchRouter
from .types import SearchArgs, SearchResponse, SearchResult

__all__ = [
    "EPHEMERAL_WEB_SEARCH_CONTEXT_MARKER",
    "SearchArgs",
    "SearchConfig",
    "SearchProviderConfig",
    "SearchQueryRefinement",
    "SearchQueryRefiner",
    "SearchQueryRefinerConfig",
    "SearchResponse",
    "SearchResult",
    "SearchReranker",
    "SearchRerankerConfig",
    "SearchReranking",
    "SearchRouter",
    "build_search_router",
    "wrap_search_results",
]
