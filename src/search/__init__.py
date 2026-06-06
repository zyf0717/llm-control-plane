"""Lightweight search discovery package."""

from .index import build_search_router
from .safety import wrap_search_results
from .search_router import SearchConfig, SearchProviderConfig, SearchRouter
from .types import SearchArgs, SearchResponse, SearchResult

__all__ = [
    "SearchArgs",
    "SearchConfig",
    "SearchProviderConfig",
    "SearchResponse",
    "SearchResult",
    "SearchRouter",
    "build_search_router",
    "wrap_search_results",
]
