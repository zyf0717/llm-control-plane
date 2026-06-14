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

    with patch(
        "src.orchestrator.proxy_services.search_service.search",
        AsyncMock(return_value=response),
    ) as search:
        result = client.post("/search/web", json={"query": "alpha", "count": 5})

    assert result.status_code == 200
    body = result.json()
    assert body["query"] == "alpha"
    assert body["provider"] == "duckduckgo_html"
    assert body["results"][0]["title"] == "Alpha"
    assert search.await_args.args[0].use_reranker is False


def test_search_web_endpoint_rejects_missing_query():
    result = client.post("/search/web", json={})

    assert result.status_code == 400
    assert result.json()["detail"] == "query is required"


def test_search_web_endpoint_accepts_source_text():
    response = SearchResponse(query="alpha", provider="none", results=[], warnings=[])

    with patch(
        "src.orchestrator.proxy_services.search_service.search",
        AsyncMock(return_value=response),
    ) as search:
        result = client.post(
            "/search/web",
            json={"query": "alpha", "source_text": "prior source_text"},
        )

    assert result.status_code == 200
    assert search.await_args.args[0].source_text == "prior source_text"
    assert "search_evidence" in result.json()


def test_search_web_endpoint_accepts_new_query_refiner_bypass_flag():
    response = SearchResponse(query="alpha", provider="none", results=[], warnings=[])

    with patch(
        "src.orchestrator.proxy_services.search_service.search",
        AsyncMock(return_value=response),
    ) as search:
        result = client.post(
            "/search/web",
            json={"query": "alpha", "use_query_refiner": False},
        )

    assert result.status_code == 200
    assert search.await_args.args[0].use_query_refiner is False


def test_search_web_endpoint_accepts_reranker_bypass_flag():
    response = SearchResponse(query="alpha", provider="none", results=[], warnings=[])

    with patch(
        "src.orchestrator.proxy_services.search_service.search",
        AsyncMock(return_value=response),
    ) as search:
        result = client.post(
            "/search/web",
            json={"query": "alpha", "use_reranker": False},
        )

    assert result.status_code == 200
    assert search.await_args.args[0].use_reranker is False


def test_search_web_endpoint_accepts_rerank_source_text():
    response = SearchResponse(query="alpha", provider="none", results=[], warnings=[])

    with patch(
        "src.orchestrator.proxy_services.search_service.search",
        AsyncMock(return_value=response),
    ) as search:
        result = client.post(
            "/search/web",
            json={
                "query": "alpha",
                "useReranker": False,
                "rerank_source_text": "ranking source text",
            },
        )

    assert result.status_code == 200
    assert search.await_args.args[0].use_reranker is False
    assert search.await_args.args[0].rerank_source_text == "ranking source text"


def test_search_web_endpoint_rejects_old_context_fields():
    result = client.post(
        "/search/web",
        json={
            "query": "alpha",
            "context": "old",
            "rerankContext": "old",
        },
    )

    assert result.status_code == 400
    assert "Unsupported search field" in result.json()["detail"]


def test_search_web_endpoint_ignores_legacy_planner_bypass_flag():
    response = SearchResponse(query="alpha", provider="none", results=[], warnings=[])

    with patch(
        "src.orchestrator.proxy_services.search_service.search",
        AsyncMock(return_value=response),
    ) as search:
        result = client.post(
            "/search/web",
            json={"query": "alpha", "usePlanner": False},
        )

    assert result.status_code == 200
    assert search.await_args.args[0].use_query_refiner is True


def test_search_web_endpoint_uses_new_query_refiner_flag_without_legacy_precedence():
    response = SearchResponse(query="alpha", provider="none", results=[], warnings=[])

    with patch(
        "src.orchestrator.proxy_services.search_service.search",
        AsyncMock(return_value=response),
    ) as search:
        result = client.post(
            "/search/web",
            json={
                "query": "alpha",
                "use_query_refiner": False,
                "usePlanner": True,
            },
        )

    assert result.status_code == 200
    assert search.await_args.args[0].use_query_refiner is False


def test_search_web_endpoint_emits_query_refinement_without_planner_alias():
    response = SearchResponse(
        query="refined",
        provider="none",
        results=[],
        warnings=[],
        original_query="original",
        query_refinement={"used": True, "effective_query": "refined"},
    )

    with patch(
        "src.orchestrator.proxy_services.search_service.search",
        AsyncMock(return_value=response),
    ):
        result = client.post("/search/web", json={"query": "original"})

    assert result.status_code == 200
    body = result.json()
    assert body["query_refinement"] == {"used": True, "effective_query": "refined"}
    assert "planner" not in body


def test_search_web_endpoint_emits_reranking_metadata():
    response = SearchResponse(
        query="alpha",
        provider="none",
        results=[],
        warnings=[],
        reranking={"used": True, "model": "search-reranker", "path": "llm"},
    )

    with patch(
        "src.orchestrator.proxy_services.search_service.search",
        AsyncMock(return_value=response),
    ):
        result = client.post("/search/web", json={"query": "alpha"})

    assert result.status_code == 200
    assert result.json()["reranking"] == {
        "used": True,
        "model": "search-reranker",
        "path": "llm",
    }
