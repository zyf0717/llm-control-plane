import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.search.reranker import SearchReranker, SearchRerankerConfig
from src.search.types import SearchResult


class FakeRerankerResponse:
    def __init__(self, content: str):
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self.content}}]}


def _result(title: str, rank: int) -> SearchResult:
    return SearchResult(
        title=title,
        url=f"https://example.com/{rank}",
        snippet=f"snippet {rank}",
        rank=rank,
        provider="duckduckgo_html",
        engine="DuckDuckGo",
        fetched_at="2026-06-06T00:00:00+00:00",
    )


@pytest.mark.asyncio
async def test_reranker_disabled_returns_original_order():
    reranker = SearchReranker(SearchRerankerConfig(enabled=False))
    results = [_result("A", 1), _result("B", 2)]

    ranking = await reranker.rerank(query="q", results=results)

    assert [result.title for result in ranking.results] == ["A", "B"]
    assert ranking.used is False
    assert ranking.degraded is False


@pytest.mark.asyncio
async def test_valid_reranker_response_reorders_results_and_attaches_metadata():
    payload = json.dumps(
        {
            "ranked": [
                {"id": "2", "score": 0.9, "reason": "most relevant"},
                {"id": "1", "score": 0.2, "reason": "less direct"},
            ]
        }
    )
    reranker = SearchReranker(
        SearchRerankerConfig(enabled=True, model_endpoint="https://reranker.local")
    )
    results = [_result("A", 1), _result("B", 2)]

    with patch(
        "httpx.AsyncClient.post",
        AsyncMock(return_value=FakeRerankerResponse(payload)),
    ) as post:
        ranking = await reranker.rerank(
            query="q",
            results=results,
            context="prior context",
        )

    assert post.await_args.args[0] == "https://reranker.local/v1/chat/completions"
    assert [result.title for result in ranking.results] == ["B", "A"]
    assert [result.rank for result in ranking.results] == [1, 2]
    assert ranking.results[0].score == 0.9
    assert ranking.results[0].ranking == {
        "reranker": "search-reranker",
        "reason": "most relevant",
    }
    assert ranking.to_public_dict() == {
        "used": True,
        "degraded": False,
        "model": "search-reranker",
    }


@pytest.mark.asyncio
async def test_reranker_ignores_unknown_duplicate_and_appends_omitted_candidates():
    payload = json.dumps(
        {
            "ranked": [
                {"id": "3", "score": 0.8},
                {"id": "missing", "score": 1},
                {"id": "3", "score": 0.1},
            ]
        }
    )
    reranker = SearchReranker(
        SearchRerankerConfig(enabled=True, model_endpoint="https://reranker.local")
    )
    results = [_result("A", 1), _result("B", 2), _result("C", 3)]

    with patch(
        "httpx.AsyncClient.post",
        AsyncMock(return_value=FakeRerankerResponse(payload)),
    ):
        ranking = await reranker.rerank(query="q", results=results)

    assert [result.title for result in ranking.results] == ["C", "A", "B"]
    assert [result.rank for result in ranking.results] == [1, 2, 3]


@pytest.mark.asyncio
async def test_invalid_json_falls_back_to_original_order():
    reranker = SearchReranker(
        SearchRerankerConfig(enabled=True, model_endpoint="https://reranker.local")
    )
    results = [_result("A", 1), _result("B", 2)]

    with patch(
        "httpx.AsyncClient.post",
        AsyncMock(return_value=FakeRerankerResponse("not json")),
    ):
        ranking = await reranker.rerank(query="q", results=results)

    assert [result.title for result in ranking.results] == ["A", "B"]
    assert ranking.degraded is True
    assert ranking.warning == "reranker-failed: JSONDecodeError"


@pytest.mark.asyncio
async def test_empty_ranking_falls_back_to_original_order():
    reranker = SearchReranker(
        SearchRerankerConfig(enabled=True, model_endpoint="https://reranker.local")
    )
    results = [_result("A", 1), _result("B", 2)]

    with patch(
        "httpx.AsyncClient.post",
        AsyncMock(return_value=FakeRerankerResponse('{"ranked": []}')),
    ):
        ranking = await reranker.rerank(query="q", results=results)

    assert [result.title for result in ranking.results] == ["A", "B"]
    assert ranking.degraded is True
    assert ranking.warning == "empty-ranking"


@pytest.mark.asyncio
async def test_timeout_falls_back_to_original_order():
    reranker = SearchReranker(
        SearchRerankerConfig(enabled=True, model_endpoint="https://reranker.local")
    )
    results = [_result("A", 1), _result("B", 2)]

    with patch(
        "httpx.AsyncClient.post",
        AsyncMock(side_effect=httpx.TimeoutException("timeout")),
    ):
        ranking = await reranker.rerank(query="q", results=results)

    assert [result.title for result in ranking.results] == ["A", "B"]
    assert ranking.degraded is True
    assert ranking.warning == "reranker-failed: TimeoutException"


@pytest.mark.asyncio
async def test_context_and_candidate_limits_are_enforced():
    reranker = SearchReranker(
        SearchRerankerConfig(
            enabled=True,
            model_endpoint="https://reranker.local",
            max_context_chars=4,
            max_candidates=1,
        )
    )
    results = [_result("A", 1), _result("B", 2)]

    with patch(
        "httpx.AsyncClient.post",
        AsyncMock(return_value=FakeRerankerResponse('{"ranked": [{"id": "1"}]}')),
    ) as post:
        ranking = await reranker.rerank(
            query="q",
            results=results,
            context="abcdef",
        )

    body = post.await_args.kwargs["json"]
    user_payload = json.loads(body["messages"][1]["content"])
    assert user_payload["context"] == "abcd"
    assert len(user_payload["candidates"]) == 1
    assert [result.title for result in ranking.results] == ["A", "B"]


@pytest.mark.asyncio
async def test_public_metadata_does_not_expose_prompt_or_raw_output():
    reranker = SearchReranker(
        SearchRerankerConfig(enabled=True, model_endpoint="https://reranker.local")
    )

    with patch(
        "httpx.AsyncClient.post",
        AsyncMock(
            return_value=FakeRerankerResponse('```json\n{"ranked":[{"id":"1"}]}\n```')
        ),
    ):
        ranking = await reranker.rerank(query="q", results=[_result("A", 1)])

    public = ranking.to_public_dict()
    assert public["used"] is True
    assert "messages" not in public
    assert "raw" not in public
