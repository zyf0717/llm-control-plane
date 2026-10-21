from unittest.mock import AsyncMock

from src.search import build_search_router


def test_build_search_router_injects_one_utility_completion():
    complete = AsyncMock()
    router = build_search_router(
        {
            "query_refiner_enabled": True,
            "reranker_enabled": True,
            "reranker_endpoint": "https://dedicated.local",
            "reranker_backend": "dedicated",
        },
        llm_complete=complete,
    )
    assert router.query_refiner.llm_complete is complete
    assert router.reranker.llm_complete is complete
    assert router.reranker.config.endpoint == "https://dedicated.local"


def test_build_search_router_helpers_disabled_by_default():
    router = build_search_router({"enabled": True})
    assert router.query_refiner is None
    assert router.reranker is None


def test_build_search_router_keeps_query_refiner_bounds():
    router = build_search_router(
        {
            "query_refiner_enabled": True,
            "query_refiner_timeout_ms": 222,
            "query_refiner_max_source_chars": 444,
            "query_refiner_max_output_tokens": 666,
            "query_refiner_max_queries": 3,
        },
        llm_complete=AsyncMock(),
    )
    config = router.query_refiner.config
    assert (
        config.timeout_ms,
        config.max_source_chars,
        config.max_output_tokens,
        config.max_queries,
    ) == (222, 444, 666, 3)


def test_build_search_router_keeps_dedicated_reranker_options():
    headers = {"CF-Access-Client-Id": "id"}
    router = build_search_router(
        {
            "reranker_endpoint": "https://dedicated.local",
            "reranker_backend": "dedicated",
            "reranker_model": "reranker-model",
            "reranker_timeout_ms": 333,
            "reranker_max_source_chars": 555,
            "reranker_max_candidates": 7,
            "reranker_max_output_tokens": 777,
        },
        reranker_headers=headers,
    )
    headers["CF-Access-Client-Id"] = "changed"
    config = router.reranker.config
    assert config.model == "reranker-model"
    assert (
        config.timeout_ms,
        config.max_source_chars,
        config.max_candidates,
        config.max_output_tokens,
    ) == (333, 555, 7, 777)
    assert config.headers == {"CF-Access-Client-Id": "id"}


def test_build_search_router_defaults_reranker_backend_to_llm():
    router = build_search_router({"reranker_enabled": True}, llm_complete=AsyncMock())
    assert router.reranker.config.backend == "llm"
