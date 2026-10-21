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
        return {
            "model": "served-reranker",
            "endpoint": "utility-node",
            "choices": [{"message": {"content": self.content}}],
        }


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
async def test_dedicated_reranker_calls_utility_only_on_failure():
    complete = AsyncMock(
        return_value=FakeRerankerResponse('{"ranked":[{"id":"2"},{"id":"1"}]}').json()
    )
    reranker = SearchReranker(
        SearchRerankerConfig(
            enabled=True, backend="dedicated", endpoint="https://dedicated.local"
        ),
        llm_complete=complete,
    )
    dedicated = FakeRerankerResponse(
        payload={"results": [{"index": 1, "relevance_score": 0.9}]}
    )
    with patch(
        "httpx.AsyncClient.post",
        AsyncMock(side_effect=[dedicated, httpx.ConnectError("offline")]),
    ):
        first = await reranker.rerank(
            query="q", results=[_result("A", 1), _result("B", 2)]
        )
        complete.assert_not_awaited()
        second = await reranker.rerank(
            query="q", results=[_result("A", 1), _result("B", 2)]
        )
    assert first.path == "dedicated"
    assert second.path == "llm" and second.used and second.degraded
    assert [r.title for r in second.results] == ["B", "A"]
    complete.assert_awaited_once()
    assert complete.await_args.args[0]["reasoning_effort"] == "none"
    assert "model" not in complete.await_args.args[0]


@pytest.mark.asyncio
async def test_llm_reranker_preserves_order_if_utility_unavailable():
    complete = AsyncMock(
        side_effect=[
            FakeRerankerResponse('{"ranked":[{"id":"2"},{"id":"1"}]}').json(),
            RuntimeError("offline"),
        ]
    )
    reranker = SearchReranker(SearchRerankerConfig(enabled=True), llm_complete=complete)
    ranked = await reranker.rerank(
        query="q", results=[_result("A", 1), _result("B", 2)]
    )
    offline = await reranker.rerank(
        query="q", results=[_result("A", 1), _result("B", 2)]
    )
    assert ranked.used and [r.title for r in ranked.results] == ["B", "A"]
    assert not offline.used and offline.degraded
    assert [r.title for r in offline.results] == ["A", "B"]


@pytest.mark.asyncio
async def test_reranker_disabled_returns_original_order():
    reranker = SearchReranker(
        SearchRerankerConfig(enabled=False), llm_complete=AsyncMock()
    )
    results = [_result("A", 1), _result("B", 2)]

    ranking = await reranker.rerank(query="q", results=results)

    assert [result.title for result in ranking.results] == ["A", "B"]
    assert ranking.used is False
    assert ranking.degraded is False


@pytest.mark.asyncio
async def test_reranker_disabled_honors_top_k_limit():
    reranker = SearchReranker(
        SearchRerankerConfig(enabled=False), llm_complete=AsyncMock()
    )
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
        SearchRerankerConfig(enabled=True), llm_complete=AsyncMock()
    )
    results = [_result("A", 1), _result("B", 2)]

    with patch.object(
        reranker,
        "llm_complete",
        AsyncMock(return_value=FakeRerankerResponse(payload).json()),
    ):
        ranking = await reranker.rerank(
            query="q",
            results=results,
            source_text="prior source_text",
        )

    assert [result.title for result in ranking.results] == ["B", "A"]
    assert [result.rank for result in ranking.results] == [1, 2]
    assert ranking.results[0].score == 0.9
    assert ranking.results[0].ranking == {
        "reranker": "served-reranker",
        "reranker_path": "llm",
        "reason": "most relevant",
    }
    assert ranking.results[1].ranking["reranker_path"] == "llm"
    assert ranking.to_public_dict() == {
        "used": True,
        "degraded": False,
        "model": "served-reranker",
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
        SearchRerankerConfig(enabled=True), llm_complete=AsyncMock()
    )
    results = [_result("A", 1), _result("B", 2), _result("C", 3)]

    with patch.object(
        reranker,
        "llm_complete",
        AsyncMock(return_value=FakeRerankerResponse(payload).json()),
    ):
        ranking = await reranker.rerank(query="q", results=results)

    assert [result.title for result in ranking.results] == ["C", "A", "B"]
    assert [result.rank for result in ranking.results] == [1, 2, 3]


@pytest.mark.asyncio
async def test_invalid_json_falls_back_to_original_order():
    reranker = SearchReranker(
        SearchRerankerConfig(enabled=True), llm_complete=AsyncMock()
    )
    results = [_result("A", 1), _result("B", 2)]

    with patch.object(
        reranker,
        "llm_complete",
        AsyncMock(return_value=FakeRerankerResponse("not json").json()),
    ):
        ranking = await reranker.rerank(query="q", results=results)

    assert [result.title for result in ranking.results] == ["A", "B"]
    assert ranking.degraded is True
    assert ranking.warning == "reranker-failed: JSONDecodeError"


@pytest.mark.asyncio
async def test_mixed_prose_json_reranker_response_is_rejected():
    reranker = SearchReranker(
        SearchRerankerConfig(enabled=True), llm_complete=AsyncMock()
    )
    results = [_result("A", 1), _result("B", 2)]

    with patch.object(
        reranker,
        "llm_complete",
        AsyncMock(
            return_value=FakeRerankerResponse(
                'Here is JSON:\n{"ranked": [{"id": "2"}]}'
            ).json()
        ),
    ):
        ranking = await reranker.rerank(query="q", results=results)

    assert [result.title for result in ranking.results] == ["A", "B"]
    assert ranking.degraded is True
    assert ranking.warning == "reranker-failed: JSONDecodeError"


@pytest.mark.asyncio
async def test_schema_invalid_reranker_response_falls_back_to_original_order():
    reranker = SearchReranker(
        SearchRerankerConfig(enabled=True), llm_complete=AsyncMock()
    )
    results = [_result("A", 1), _result("B", 2)]

    with patch.object(
        reranker,
        "llm_complete",
        AsyncMock(
            return_value=FakeRerankerResponse('{"ranked": [{"score": 0.9}]}').json()
        ),
    ):
        ranking = await reranker.rerank(query="q", results=results)

    assert [result.title for result in ranking.results] == ["A", "B"]
    assert ranking.degraded is True
    assert ranking.warning == "reranker-failed: ValueError"


@pytest.mark.asyncio
async def test_empty_ranking_falls_back_to_original_order():
    reranker = SearchReranker(
        SearchRerankerConfig(enabled=True), llm_complete=AsyncMock()
    )
    results = [_result("A", 1), _result("B", 2)]

    with patch.object(
        reranker,
        "llm_complete",
        AsyncMock(return_value=FakeRerankerResponse('{"ranked": []}').json()),
    ):
        ranking = await reranker.rerank(query="q", results=results)

    assert [result.title for result in ranking.results] == ["A", "B"]
    assert ranking.degraded is True
    assert ranking.warning == "empty-ranking"


@pytest.mark.asyncio
async def test_timeout_falls_back_to_original_order():
    reranker = SearchReranker(
        SearchRerankerConfig(enabled=True), llm_complete=AsyncMock()
    )
    results = [_result("A", 1), _result("B", 2)]

    with patch.object(
        reranker,
        "llm_complete",
        AsyncMock(side_effect=httpx.TimeoutException("timeout")),
    ):
        ranking = await reranker.rerank(query="q", results=results)

    assert [result.title for result in ranking.results] == ["A", "B"]
    assert ranking.degraded is True
    assert ranking.warning == "reranker-failed: TimeoutException"


@pytest.mark.asyncio
async def test_context_and_candidate_limits_are_enforced():
    reranker = SearchReranker(
        SearchRerankerConfig(enabled=True, max_source_chars=4, max_candidates=1),
        llm_complete=AsyncMock(),
    )
    results = [_result("A", 1), _result("B", 2)]

    with patch.object(
        reranker,
        "llm_complete",
        AsyncMock(
            return_value=FakeRerankerResponse('{"ranked": [{"id": "1"}]}').json()
        ),
    ) as post:
        ranking = await reranker.rerank(
            query="q",
            results=results,
            source_text="abcdef",
        )

    body = post.await_args.args[0]
    user_payload = json.loads(body["messages"][1]["content"])
    assert user_payload["context"] == "abcd"
    assert len(user_payload["candidates"]) == 1
    assert [result.title for result in ranking.results] == ["A", "B"]


@pytest.mark.asyncio
async def test_public_metadata_does_not_expose_prompt_or_raw_output():
    reranker = SearchReranker(
        SearchRerankerConfig(enabled=True), llm_complete=AsyncMock()
    )

    with patch.object(
        reranker,
        "llm_complete",
        AsyncMock(
            return_value=FakeRerankerResponse(
                '```json\n{"ranked":[{"id":"1"}]}\n```'
            ).json()
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
            endpoint="https://reranker.local",
            backend="dedicated",
            max_candidates=2,
        ),
        llm_complete=AsyncMock(),
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
        "backend": "dedicated",
        "path": "dedicated",
    }
    assert ranking.results[0].ranking["reranker_path"] == "dedicated"


@pytest.mark.asyncio
async def test_dedicated_reranker_top_k_caps_returned_results():
    reranker = SearchReranker(
        SearchRerankerConfig(
            enabled=True,
            endpoint="https://reranker.local",
            backend="dedicated",
            max_candidates=3,
        ),
        llm_complete=AsyncMock(),
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
            enabled=True, endpoint="https://reranker.local", backend="dedicated"
        ),
        llm_complete=AsyncMock(),
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
async def test_dedicated_and_llm_fallback_fail_preserves_provider_order():
    complete = AsyncMock(side_effect=httpx.ConnectError("connect"))
    reranker = SearchReranker(
        SearchRerankerConfig(
            enabled=True, endpoint="https://dedicated.local", backend="dedicated"
        ),
        llm_complete=complete,
    )
    with patch(
        "httpx.AsyncClient.post",
        AsyncMock(side_effect=httpx.TimeoutException("timeout")),
    ):
        ranking = await reranker.rerank(
            query="q", results=[_result("A", 1), _result("B", 2)]
        )
    assert [r.title for r in ranking.results] == ["A", "B"]
    assert ranking.degraded
    assert (
        ranking.warning
        == "dedicated-reranker-failed: TimeoutException; llm-fallback-failed: ConnectError"
    )
    assert ranking.path == "none" and ranking.backend == "dedicated"
