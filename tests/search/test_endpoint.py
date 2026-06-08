from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from src.orchestrator.proxy import app
from src.search.types import SearchResponse, SearchResult


client = TestClient(app)


def test_search_web_endpoint_returns_results():
    response = SearchResponse(
        query="alpha",
        provider="duckduckgo_html",
        results=[
            SearchResult(
                title="Alpha",
                url="https://example.com/a",
                snippet="snippet",
                rank=1,
                provider="duckduckgo_html",
                engine="DuckDuckGo",
                fetched_at="2026-06-06T00:00:00+00:00",
            )
        ],
        warnings=[],
    )

    with patch("src.orchestrator.proxy.search_service.search", AsyncMock(return_value=response)):
        result = client.post("/search/web", json={"query": "alpha", "count": 5})

    assert result.status_code == 200
    body = result.json()
    assert body["query"] == "alpha"
    assert body["provider"] == "duckduckgo_html"
    assert body["results"][0]["title"] == "Alpha"


def test_search_web_endpoint_rejects_missing_query():
    result = client.post("/search/web", json={})

    assert result.status_code == 400
    assert result.json()["detail"] == "query is required"


def test_search_web_endpoint_accepts_context():
    response = SearchResponse(query="alpha", provider="none", results=[], warnings=[])

    with patch(
        "src.orchestrator.proxy.search_service.search",
        AsyncMock(return_value=response),
    ) as search:
        result = client.post(
            "/search/web",
            json={"query": "alpha", "context": "prior context"},
        )

    assert result.status_code == 200
    assert search.await_args.args[0].context == "prior context"
    assert "wrapped_results" in result.json()
