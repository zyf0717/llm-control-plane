import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.search.reranker import SearchReranker, SearchRerankerConfig
from src.search.types import SearchResult


class FakeRerankerResponse:
    def __init__(self, content: str = "", payload=None):
        self.content = content
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        if self.payload is not None:
            return self.payload
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
async def test_reranker_disabled_honors_top_k_limit():
    reranker = SearchReranker(SearchRerankerConfig(enabled=False))
    results = [_result("A", 1), _result("B", 2)]

    ranking = await reranker.rerank(query="q", results=results, top_k=1)

    assert [result.title for result in ranking.results] == ["A"]
    assert [result.rank for result in ranking.results] == [1]
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
            source_text="prior source_text",
        )

    assert post.await_args.args[0] == "https://reranker.local/v1/chat/completions"
    assert [result.title for result in ranking.results] == ["B", "A"]
    assert [result.rank for result in ranking.results] == [1, 2]
    assert ranking.results[0].score == 0.9
    assert ranking.results[0].ranking == {
        "reranker": "search-reranker",
        "reranker_path": "llm",
        "reason": "most relevant",
    }
    assert ranking.results[1].ranking["reranker_path"] == "llm"
    assert ranking.to_public_dict() == {
        "used": True,
        "degraded": False,
        "model": "search-reranker",
        "backend": "llm",
        "path": "llm",
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
async def test_mixed_prose_json_reranker_response_is_rejected():
    reranker = SearchReranker(
        SearchRerankerConfig(enabled=True, model_endpoint="https://reranker.local")
    )
    results = [_result("A", 1), _result("B", 2)]

    with patch(
        "httpx.AsyncClient.post",
        AsyncMock(
            return_value=FakeRerankerResponse(
                'Here is JSON:\n{"ranked": [{"id": "2"}]}'
            )
        ),
    ):
        ranking = await reranker.rerank(query="q", results=results)

    assert [result.title for result in ranking.results] == ["A", "B"]
    assert ranking.degraded is True
    assert ranking.warning == "reranker-failed: JSONDecodeError"


@pytest.mark.asyncio
async def test_schema_invalid_reranker_response_falls_back_to_original_order():
    reranker = SearchReranker(
        SearchRerankerConfig(enabled=True, model_endpoint="https://reranker.local")
    )
    results = [_result("A", 1), _result("B", 2)]

    with patch(
        "httpx.AsyncClient.post",
        AsyncMock(return_value=FakeRerankerResponse('{"ranked": [{"score": 0.9}]}')),
    ):
        ranking = await reranker.rerank(query="q", results=results)

    assert [result.title for result in ranking.results] == ["A", "B"]
    assert ranking.degraded is True
    assert ranking.warning == "reranker-failed: ValueError"


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
            max_source_chars=4,
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
            source_text="abcdef",
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


@pytest.mark.asyncio
async def test_dedicated_reranker_posts_documents_and_reorders_by_index():
    reranker = SearchReranker(
        SearchRerankerConfig(
            enabled=True,
            model_endpoint="https://reranker.local",
            backend="dedicated",
            max_candidates=2,
        )
    )
    results = [_result("A", 1), _result("B", 2), _result("C", 3)]
    response = {
        "results": [
            {"index": 1, "relevance_score": 1.2},
            {"index": 0, "similarity": -0.1},
        ]
    }

    with patch(
        "httpx.AsyncClient.post",
        AsyncMock(return_value=FakeRerankerResponse(payload=response)),
    ) as post:
        ranking = await reranker.rerank(
            query="q",
            results=results,
            source_text="abcdef",
        )

    assert post.await_args.args[0] == "https://reranker.local/rerank"
    body = post.await_args.kwargs["json"]
    assert body == {
        "query": "q\n\nContext:\nabcdef",
        "documents": [
            "A\nsnippet 1\nhttps://example.com/1",
            "B\nsnippet 2\nhttps://example.com/2",
        ],
        "top_k": 2,
    }
    assert [result.title for result in ranking.results] == ["B", "A", "C"]
    assert [result.rank for result in ranking.results] == [1, 2, 3]
    assert ranking.results[0].score == 1.0
    assert ranking.results[1].score == 0.0
    assert ranking.to_public_dict() == {
        "used": True,
        "degraded": False,
        "model": "search-reranker",
        "backend": "dedicated",
        "path": "dedicated",
    }
    assert ranking.results[0].ranking["reranker_path"] == "dedicated"


@pytest.mark.asyncio
async def test_dedicated_reranker_top_k_caps_returned_results():
    reranker = SearchReranker(
        SearchRerankerConfig(
            enabled=True,
            model_endpoint="https://reranker.local",
            backend="dedicated",
            max_candidates=3,
        )
    )
    results = [_result("A", 1), _result("B", 2), _result("C", 3)]
    response = {
        "results": [
            {"index": 2, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.8},
            {"index": 0, "relevance_score": 0.7},
        ]
    }

    with patch(
        "httpx.AsyncClient.post",
        AsyncMock(return_value=FakeRerankerResponse(payload=response)),
    ) as post:
        ranking = await reranker.rerank(query="q", results=results, top_k=2)

    assert post.await_args.kwargs["json"]["top_k"] == 2
    assert [result.title for result in ranking.results] == ["C", "B"]
    assert [result.rank for result in ranking.results] == [1, 2]


@pytest.mark.asyncio
async def test_dedicated_reranker_ignores_unknown_duplicate_and_appends_omitted():
    reranker = SearchReranker(
        SearchRerankerConfig(
            enabled=True,
            model_endpoint="https://reranker.local",
            backend="dedicated",
        )
    )
    results = [_result("A", 1), _result("B", 2), _result("C", 3)]
    response = {
        "results": [
            {"id": "3", "score": 0.9},
            {"id": "missing", "score": 1},
            {"id": "3", "score": 0.1},
        ]
    }

    with patch(
        "httpx.AsyncClient.post",
        AsyncMock(return_value=FakeRerankerResponse(payload=response)),
    ):
        ranking = await reranker.rerank(query="q", results=results)

    assert [result.title for result in ranking.results] == ["C", "A", "B"]
    assert [result.rank for result in ranking.results] == [1, 2, 3]


@pytest.mark.asyncio
async def test_dedicated_reranker_falls_back_to_llm_when_configured():
    fallback_payload = json.dumps(
        {"ranked": [{"id": "2", "score": 0.8}, {"id": "1", "score": 0.3}]}
    )
    reranker = SearchReranker(
        SearchRerankerConfig(
            enabled=True,
            model_endpoint="https://dedicated.local",
            fallback_model_endpoint="https://llm.local",
            backend="dedicated",
        )
    )
    results = [_result("A", 1), _result("B", 2)]

    with patch(
        "httpx.AsyncClient.post",
        AsyncMock(
            side_effect=[
                httpx.TimeoutException("timeout"),
                FakeRerankerResponse(fallback_payload),
            ]
        ),
    ) as post:
        ranking = await reranker.rerank(query="q", results=results)

    assert post.await_args_list[0].args[0] == "https://dedicated.local/rerank"
    assert post.await_args_list[1].args[0] == "https://llm.local/v1/chat/completions"
    assert [result.title for result in ranking.results] == ["B", "A"]
    assert ranking.to_public_dict() == {
        "used": True,
        "degraded": True,
        "model": "search-reranker",
        "backend": "llm",
        "path": "llm",
        "warning": "dedicated-reranker-failed: TimeoutException",
    }


@pytest.mark.asyncio
async def test_dedicated_and_llm_fallback_fail_preserves_provider_order():
    reranker = SearchReranker(
        SearchRerankerConfig(
            enabled=True,
            model_endpoint="https://dedicated.local",
            fallback_model_endpoint="https://llm.local",
            backend="dedicated",
        )
    )
    results = [_result("A", 1), _result("B", 2)]

    with patch(
        "httpx.AsyncClient.post",
        AsyncMock(
            side_effect=[
                httpx.TimeoutException("timeout"),
                httpx.ConnectError("connect"),
            ]
        ),
    ):
        ranking = await reranker.rerank(query="q", results=results)

    assert [result.title for result in ranking.results] == ["A", "B"]
    assert ranking.degraded is True
    assert ranking.warning == (
        "dedicated-reranker-failed: TimeoutException; "
        "llm-fallback-failed: ConnectError"
    )
    assert ranking.to_public_dict()["backend"] == "dedicated"
    assert ranking.to_public_dict()["path"] == "none"
