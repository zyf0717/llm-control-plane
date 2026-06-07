"""In-memory TTL cache for parsed search responses."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from .types import SearchResponse


@dataclass(slots=True)
class _CacheEntry:
    response: SearchResponse
    expires_at: float


class SearchCache:
    """Small in-memory TTL cache keyed by normalized search request."""

    def __init__(self, default_ttl_seconds: int = 900):
        self.default_ttl_seconds = max(0, int(default_ttl_seconds))
        self._entries: dict[str, _CacheEntry] = {}

    def get(self, key: str) -> Optional[SearchResponse]:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if time.monotonic() >= entry.expires_at:
            self._entries.pop(key, None)
            return None
        return entry.response

    def set(
        self, key: str, response: SearchResponse, ttl_seconds: Optional[int] = None
    ) -> None:
        ttl = self.default_ttl_seconds if ttl_seconds is None else max(0, ttl_seconds)
        self._entries[key] = _CacheEntry(
            response=response, expires_at=time.monotonic() + ttl
        )
