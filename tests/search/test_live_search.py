import os

import pytest

from src.search import SearchArgs, build_search_router


LIVE_TEST_FLAG = "SEARCH_LIVE_TESTS"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_duckduckgo_html_live_search_returns_candidates():
    if os.getenv(LIVE_TEST_FLAG) != "1":
        pytest.skip(f"set {LIVE_TEST_FLAG}=1 to run live search tests")

    router = build_search_router(
        {
            "enabled": True,
            "default_provider": "duckduckgo_html",
            "timeout_ms": 7000,
            "max_results": 5,
            "cache_ttl_seconds": 60,
            "providers": {
                "duckduckgo_html": {
                    "enabled": True,
                    "priority": 10,
                },
                "marginalia_html": {"enabled": False},
                "mojeek_html": {"enabled": False},
                "wikipedia_opensearch": {"enabled": False},
            },
        }
    )

    try:
        response = await router.search(
            SearchArgs(
                query="site:wikipedia.org Ada Lovelace",
                provider="duckduckgo_html",
                count=5,
            )
        )
    except ValueError as exc:
        if "challenge page detected" in str(exc):
            pytest.skip(f"live provider returned bot challenge: {exc}")
        raise

    assert response.provider == "duckduckgo_html"
    assert response.degraded is False
    assert response.results
    assert all(result.url.startswith("http") for result in response.results)
    assert all(result.title for result in response.results)
