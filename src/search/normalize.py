"""Normalization helpers for provider outputs."""

from __future__ import annotations

from typing import Iterable, Optional
from urllib.parse import parse_qs, unquote, urlparse

from .types import SearchResult


def clean_search_url(raw_url: str) -> str:
    """Remove common search redirect wrappers and normalize scheme."""
    candidate = (raw_url or "").strip()
    if not candidate:
        return ""

    if candidate.startswith("//"):
        candidate = f"https:{candidate}"

    parsed = urlparse(candidate)
    query = parse_qs(parsed.query)
    redirect_target = query.get("uddg")
    if redirect_target:
        return unquote(redirect_target[0]).strip()

    return candidate


def dedupe_results(
    results: Iterable[SearchResult], max_results: Optional[int] = None
) -> list[SearchResult]:
    """Drop duplicate URLs, preserving first occurrence and compact ranks."""
    deduped: list[SearchResult] = []
    seen: set[str] = set()

    for result in results:
        key = _normalize_dedupe_key(result.url)
        if not key or key in seen:
            continue
        seen.add(key)
        result.rank = len(deduped) + 1
        deduped.append(result)
        if max_results is not None and len(deduped) >= max_results:
            break

    return deduped


def normalize_snippet(snippet: Optional[str]) -> Optional[str]:
    """Collapse blank snippets to None."""
    if snippet is None:
        return None
    normalized = " ".join(snippet.split()).strip()
    return normalized or None


def _normalize_dedupe_key(url: str) -> str:
    candidate = (url or "").strip()
    if candidate.endswith("/"):
        candidate = candidate[:-1]
    return candidate
