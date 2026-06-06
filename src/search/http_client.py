"""Bounded HTTP client for single-request search fetches."""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from .types import SearchRequest

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 compatible lightweight-search-client",
    "Accept": (
        "text/html,application/xhtml+xml,application/xml,"
        "application/json,text/plain;q=0.9,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass(slots=True)
class SearchHttpClientConfig:
    timeout_ms: int = 7000
    max_body_bytes: int = 2_500_000
    max_retries: int = 1
    headers: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_HEADERS))


class SearchHttpClient:
    """One-shot text fetcher with timeout, redirect support, and bounded retry."""

    def __init__(self, config: SearchHttpClientConfig):
        self.config = config

    async def fetch_text(self, request: SearchRequest) -> str:
        last_error: Exception | None = None
        attempts = max(1, self.config.max_retries + 1)

        for attempt in range(attempts):
            try:
                return await self._fetch_once(request)
            except (httpx.RequestError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt >= attempts - 1:
                    raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("Search HTTP fetch failed unexpectedly")

    async def _fetch_once(self, request: SearchRequest) -> str:
        headers = dict(self.config.headers)
        headers.update(request.headers)
        timeout = httpx.Timeout(self.config.timeout_ms / 1000.0)

        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
        ) as client:
            async with client.stream(
                request.method,
                request.url,
                headers=headers,
                content=request.body,
            ) as response:
                response.raise_for_status()

                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > self.config.max_body_bytes:
                        raise ValueError(
                            f"Search response exceeded {self.config.max_body_bytes} bytes"
                        )
                    chunks.append(chunk)

        payload = b"".join(chunks)
        encoding = response.encoding or "utf-8"
        return payload.decode(encoding, errors="replace")
