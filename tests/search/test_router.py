import asyncio

import pytest

from src.search.planner import SearchPlan
from src.search.search_router import SearchConfig, SearchRouter, SearchProviderConfig
from src.search.types import SearchArgs, SearchResponse, SearchResult


class StubProvider:
    def __init__(self, provider_id: str, response: SearchResponse = None, error: Exception = None):
        self.id = provider_id
        self.engine = provider_id
        self.response_type = "html"
        self.fallback_only = False
        self._response = response
        self._error = error
        self.calls = 0
        self.last_args = None

    async def search(self, args, client, cache, config, provider_config=None):
        self.calls += 1
        self.last_args = args
        if self._error is not None:
            raise self._error
        return self._response


@pytest.mark.asyncio
async def test_router_uses_first_non_empty_provider():
    empty = StubProvider(
        "empty",
        SearchResponse(query="q", provider="empty", results=[], warnings=[]),
    )
    winner = StubProvider(
        "winner",
        SearchResponse(
            query="q",
            provider="winner",
            results=[
                SearchResult(
                    title="Hit",
                    url="https://example.com",
                    snippet="snippet",
                    rank=1,
                    provider="winner",
                    engine="winner",
                    fetched_at="2026-06-06T00:00:00+00:00",
                )
            ],
            warnings=[],
        ),
    )
    router = SearchRouter(
        SearchConfig(default_provider="empty", max_results=10),
        providers={"empty": empty, "winner": winner},
        provider_configs={
            "empty": SearchProviderConfig(enabled=True, priority=10),
            "winner": SearchProviderConfig(enabled=True, priority=20),
        },
    )

    response = await router.search(SearchArgs(query="q"))

    assert response.provider == "winner"
    assert empty.calls == 1
    assert winner.calls == 1


@pytest.mark.asyncio
async def test_router_returns_degraded_when_all_providers_fail():
    router = SearchRouter(
        SearchConfig(default_provider="broken", max_results=10),
        providers={"broken": StubProvider("broken", error=RuntimeError("boom"))},
        provider_configs={"broken": SearchProviderConfig(enabled=True, priority=10)},
    )

    response = await router.search(SearchArgs(query="q"))

    assert response.provider == "none"
    assert response.degraded is True
    assert response.results == []
    assert response.warnings == ["broken: boom"]


@pytest.mark.asyncio
async def test_router_does_not_fallback_when_explicit_provider_fails():
    router = SearchRouter(
        SearchConfig(default_provider="broken", max_results=10),
        providers={
            "broken": StubProvider("broken", error=RuntimeError("boom")),
            "other": StubProvider(
                "other", SearchResponse(query="q", provider="other", results=[], warnings=[])
            ),
        },
        provider_configs={
            "broken": SearchProviderConfig(enabled=True, priority=10),
            "other": SearchProviderConfig(enabled=True, priority=20),
        },
    )

    with pytest.raises(RuntimeError, match="boom"):
        await router.search(SearchArgs(query="q", provider="broken"))


@pytest.mark.asyncio
async def test_router_rejects_unknown_explicit_provider():
    router = SearchRouter(
        SearchConfig(default_provider="broken", max_results=10),
        providers={},
        provider_configs={},
    )

    with pytest.raises(ValueError, match="Unknown search provider"):
        await router.search(SearchArgs(query="q", provider="missing"))


class StubPlanner:
    def __init__(self, plan: SearchPlan):
        self._plan = plan
        self.calls = 0
        self.last_args = None

    async def plan(self, args):
        self.calls += 1
        self.last_args = args
        return self._plan


@pytest.mark.asyncio
async def test_router_calls_planner_before_provider_and_uses_effective_query():
    provider = StubProvider(
        "winner",
        SearchResponse(
            query="planned query",
            provider="winner",
            results=[
                SearchResult(
                    title="Hit",
                    url="https://example.com",
                    snippet="snippet",
                    rank=1,
                    provider="winner",
                    engine="winner",
                    fetched_at="2026-06-06T00:00:00+00:00",
                )
            ],
        ),
    )
    planner = StubPlanner(SearchPlan(effective_query="planned query", used=True))
    router = SearchRouter(
        SearchConfig(default_provider="winner", max_results=10),
        providers={"winner": provider},
        provider_configs={"winner": SearchProviderConfig(enabled=True, priority=10)},
        planner=planner,
    )

    response = await router.search(SearchArgs(query="original query"))

    assert planner.calls == 1
    assert planner.last_args.query == "original query"
    assert provider.last_args.query == "planned query"
    assert response.original_query == "original query"
    assert response.planner["effective_query"] == "planned query"


@pytest.mark.asyncio
async def test_router_omits_original_query_when_planner_keeps_query():
    provider = StubProvider(
        "winner",
        SearchResponse(
            query="same query",
            provider="winner",
            results=[
                SearchResult(
                    title="Hit",
                    url="https://example.com",
                    snippet="snippet",
                    rank=1,
                    provider="winner",
                    engine="winner",
                    fetched_at="2026-06-06T00:00:00+00:00",
                )
            ],
        ),
    )
    planner = StubPlanner(SearchPlan(effective_query="same query", used=False))
    router = SearchRouter(
        SearchConfig(default_provider="winner", max_results=10),
        providers={"winner": provider},
        provider_configs={"winner": SearchProviderConfig(enabled=True, priority=10)},
        planner=planner,
    )

    response = await router.search(SearchArgs(query="same query"))

    assert response.original_query is None
    assert "original_query" not in response.to_dict()
    assert response.planner["used"] is False


@pytest.mark.asyncio
async def test_planner_warning_does_not_stop_provider_search():
    provider = StubProvider(
        "winner",
        SearchResponse(
            query="original",
            provider="winner",
            results=[
                SearchResult(
                    title="Hit",
                    url="https://example.com",
                    snippet="snippet",
                    rank=1,
                    provider="winner",
                    engine="winner",
                    fetched_at="2026-06-06T00:00:00+00:00",
                )
            ],
        ),
    )
    planner = StubPlanner(
        SearchPlan(
            effective_query="original",
            degraded=True,
            warning="planner-failed: TimeoutException",
        )
    )
    router = SearchRouter(
        SearchConfig(default_provider="winner", max_results=10),
        providers={"winner": provider},
        provider_configs={"winner": SearchProviderConfig(enabled=True, priority=10)},
        planner=planner,
    )

    response = await router.search(SearchArgs(query="original"))

    assert provider.calls == 1
    assert response.warnings == ["planner: planner-failed: TimeoutException"]
    assert response.planner["degraded"] is True


class FanoutProvider:
    id = "winner"
    engine = "winner"
    response_type = "html"
    fallback_only = False

    def __init__(self):
        self.calls = []
        self.active = 0
        self.max_active = 0

    async def search(self, args, client, cache, config, provider_config=None):
        self.calls.append(args.query)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01)
            urls_by_query = {
                "q1": ["https://example.com/a", "https://example.com/b"],
                "q2": ["https://example.com/b", "https://example.com/c"],
            }
            results = [
                SearchResult(
                    title=url.rsplit("/", 1)[-1],
                    url=url,
                    snippet=args.query,
                    rank=index,
                    provider=self.id,
                    engine=self.engine,
                    fetched_at="2026-06-06T00:00:00+00:00",
                )
                for index, url in enumerate(urls_by_query.get(args.query, []), start=1)
            ]
            return SearchResponse(query=args.query, provider=self.id, results=results)
        finally:
            self.active -= 1


@pytest.mark.asyncio
async def test_router_fans_out_planned_queries_and_dedupes_results():
    provider = FanoutProvider()
    planner = StubPlanner(
        SearchPlan(
            effective_query="q1",
            queries=["q1", "q2"],
            used=True,
        )
    )
    router = SearchRouter(
        SearchConfig(default_provider="winner", max_results=10, max_total_results=2),
        providers={"winner": provider},
        provider_configs={"winner": SearchProviderConfig(enabled=True, priority=10)},
        planner=planner,
    )

    response = await router.search(SearchArgs(query="original query"))

    assert sorted(provider.calls) == ["q1", "q2"]
    assert provider.max_active == 2
    assert response.query == "q1"
    assert response.provider == "winner"
    assert response.original_query == "original query"
    assert response.planner["queries"] == ["q1", "q2"]
    assert [result.url for result in response.results] == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert [result.rank for result in response.results] == [1, 2]
