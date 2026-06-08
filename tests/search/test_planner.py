import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.search.planner import SearchPlanner, SearchPlannerConfig
from src.search.types import SearchArgs


class FakePlannerResponse:
    def __init__(self, content: str):
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self.content}}]}


@pytest.mark.asyncio
async def test_planner_disabled_returns_original_query():
    planner = SearchPlanner(SearchPlannerConfig(enabled=False))

    plan = await planner.plan(SearchArgs(query="already good"))

    assert plan.effective_query == "already good"
    assert plan.used is False
    assert plan.degraded is False


@pytest.mark.asyncio
async def test_valid_planner_response_rewrites_query():
    payload = json.dumps(
        {
            "needs_search": True,
            "query": "llama.cpp KV cache metrics cached_tokens GitHub",
            "freshness": "week",
            "reason": "Software behavior may have changed.",
            "source_preferences": ["github", "official_docs"],
        }
    )
    planner = SearchPlanner(
        SearchPlannerConfig(enabled=True, model_endpoint="https://planner.local")
    )

    with patch(
        "httpx.AsyncClient.post",
        AsyncMock(return_value=FakePlannerResponse(payload)),
    ) as post:
        plan = await planner.plan(
            SearchArgs(
                query="does llama cpp expose kv cache metrics now?",
                context="prior chat",
                count=5,
                freshness="day",
            )
        )

    assert post.await_args.args[0] == "https://planner.local/v1/chat/completions"
    assert plan.effective_query == "llama.cpp KV cache metrics cached_tokens GitHub"
    assert plan.freshness == "week"
    assert plan.source_preferences == ["github", "official_docs"]
    assert plan.used is True


@pytest.mark.asyncio
async def test_planner_response_supports_capped_query_fanout():
    payload = json.dumps(
        {
            "query": "llama.cpp metrics GitHub",
            "queries": [
                "llama.cpp metrics GitHub",
                "llama.cpp release notes cached_tokens",
                "llama.cpp server metrics documentation",
                "extra query should be capped",
            ],
        }
    )
    planner = SearchPlanner(
        SearchPlannerConfig(
            enabled=True,
            model_endpoint="https://planner.local",
            max_queries=3,
            max_output_tokens=1024,
        )
    )

    with patch(
        "httpx.AsyncClient.post",
        AsyncMock(return_value=FakePlannerResponse(payload)),
    ) as post:
        plan = await planner.plan(SearchArgs(query="does llama cpp expose metrics?"))

    body = post.await_args.kwargs["json"]
    user_payload = json.loads(body["messages"][1]["content"])
    assert user_payload["max_queries"] == 3
    assert body["max_tokens"] == 1024
    assert plan.effective_query == "llama.cpp metrics GitHub"
    assert plan.queries == [
        "llama.cpp metrics GitHub",
        "llama.cpp release notes cached_tokens",
        "llama.cpp server metrics documentation",
    ]
    assert plan.to_public_dict()["queries"] == plan.queries


@pytest.mark.asyncio
async def test_planner_posts_configured_headers():
    planner = SearchPlanner(
        SearchPlannerConfig(
            enabled=True,
            model_endpoint="https://planner.local",
            headers={"CF-Access-Client-Id": "id", "CF-Access-Client-Secret": "secret"},
        )
    )

    with patch(
        "httpx.AsyncClient.post",
        AsyncMock(return_value=FakePlannerResponse('{"query": "planned"}')),
    ) as post:
        await planner.plan(SearchArgs(query="original"))

    assert post.await_args.kwargs["headers"] == {
        "CF-Access-Client-Id": "id",
        "CF-Access-Client-Secret": "secret",
    }


@pytest.mark.asyncio
async def test_invalid_json_falls_back():
    planner = SearchPlanner(
        SearchPlannerConfig(enabled=True, model_endpoint="https://planner.local")
    )

    with patch(
        "httpx.AsyncClient.post",
        AsyncMock(return_value=FakePlannerResponse("not json")),
    ):
        plan = await planner.plan(SearchArgs(query="original"))

    assert plan.effective_query == "original"
    assert plan.degraded is True
    assert plan.warning == "planner-failed: JSONDecodeError"


@pytest.mark.asyncio
async def test_timeout_falls_back():
    planner = SearchPlanner(
        SearchPlannerConfig(enabled=True, model_endpoint="https://planner.local")
    )

    with patch(
        "httpx.AsyncClient.post",
        AsyncMock(side_effect=httpx.TimeoutException("timeout")),
    ):
        plan = await planner.plan(SearchArgs(query="original"))

    assert plan.effective_query == "original"
    assert plan.degraded is True
    assert plan.warning == "planner-failed: TimeoutException"


@pytest.mark.asyncio
async def test_empty_planned_query_falls_back():
    planner = SearchPlanner(
        SearchPlannerConfig(enabled=True, model_endpoint="https://planner.local")
    )

    with patch(
        "httpx.AsyncClient.post",
        AsyncMock(return_value=FakePlannerResponse('{"query": ""}')),
    ):
        plan = await planner.plan(SearchArgs(query="original"))

    assert plan.effective_query == "original"
    assert plan.degraded is True
    assert plan.warning == "empty-query"


@pytest.mark.asyncio
async def test_long_context_is_truncated():
    planner = SearchPlanner(
        SearchPlannerConfig(
            enabled=True,
            model_endpoint="https://planner.local",
            max_context_chars=4,
        )
    )

    with patch(
        "httpx.AsyncClient.post",
        AsyncMock(return_value=FakePlannerResponse('{"query": "q"}')),
    ) as post:
        await planner.plan(SearchArgs(query="q", context="abcdef"))

    body = post.await_args.kwargs["json"]
    user_message = body["messages"][1]["content"]
    assert json.loads(user_message)["context"] == "abcd"


@pytest.mark.asyncio
async def test_source_preferences_are_sanitized():
    payload = json.dumps(
        {
            "query": "q2",
            "freshness": "invalid",
            "source_preferences": [" github ", "", 3, "x" * 100] * 4,
        }
    )
    planner = SearchPlanner(
        SearchPlannerConfig(enabled=True, model_endpoint="https://planner.local")
    )

    with patch(
        "httpx.AsyncClient.post",
        AsyncMock(return_value=FakePlannerResponse(payload)),
    ):
        plan = await planner.plan(SearchArgs(query="q"))

    assert plan.freshness is None
    assert len(plan.source_preferences) == 8
    assert plan.source_preferences[:4] == ["github", "x" * 64, "github", "x" * 64]


@pytest.mark.asyncio
async def test_public_metadata_does_not_expose_prompt_or_raw_output():
    planner = SearchPlanner(
        SearchPlannerConfig(enabled=True, model_endpoint="https://planner.local")
    )

    with patch(
        "httpx.AsyncClient.post",
        AsyncMock(return_value=FakePlannerResponse('```json\n{"query":"planned"}\n```')),
    ):
        plan = await planner.plan(SearchArgs(query="original"))

    public = plan.to_public_dict()
    assert public["effective_query"] == "planned"
    assert "messages" not in public
    assert "raw" not in public
