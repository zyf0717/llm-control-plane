from src.search import build_search_router


def test_build_search_router_injects_query_refiner_headers():
    router = build_search_router(
        {
            "enabled": True,
            "query_refiner_model_endpoint": "https://query-refiner.local",
            "providers": {"duckduckgo_html": {"enabled": True}},
        },
        query_refiner_headers={"CF-Access-Client-Id": "id"},
    )

    assert router.query_refiner is not None
    assert router.query_refiner.config.headers == {"CF-Access-Client-Id": "id"}


def test_build_search_router_does_not_alias_query_refiner_headers():
    headers = {"CF-Access-Client-Id": "id"}
    router = build_search_router(
        {"enabled": True, "query_refiner_model_endpoint": "https://query-refiner.local"},
        query_refiner_headers=headers,
    )

    headers["CF-Access-Client-Id"] = "changed"

    assert router.query_refiner is not None
    assert router.query_refiner.config.headers == {"CF-Access-Client-Id": "id"}


def test_build_search_router_ignores_legacy_query_refiner_config_keys():
    router = build_search_router(
        {
            "enabled": True,
            "model_endpoint": "https://legacy-query-refiner.local",
            "model": "legacy-model",
            "planner_enabled": True,
            "planner_timeout_ms": 123,
            "planner_max_context_chars": 456,
            "planner_max_output_tokens": 789,
            "planner_max_queries": 2,
        }
    )

    assert router.query_refiner is None


def test_build_search_router_uses_query_refiner_config_keys():
    router = build_search_router(
        {
            "enabled": True,
            "query_refiner_model_endpoint": "https://new.local",
            "query_refiner_model": "new-model",
            "query_refiner_enabled": True,
            "query_refiner_timeout_ms": 222,
            "query_refiner_max_context_chars": 444,
            "query_refiner_max_output_tokens": 666,
            "query_refiner_max_queries": 3,
        }
    )

    assert router.query_refiner is not None
    assert router.query_refiner.config.model_endpoint == "https://new.local"
    assert router.query_refiner.config.model == "new-model"
    assert router.query_refiner.config.timeout_ms == 222
    assert router.query_refiner.config.max_context_chars == 444
    assert router.query_refiner.config.max_output_tokens == 666
    assert router.query_refiner.config.max_queries == 3


def test_build_search_router_uses_reranker_config_keys():
    router = build_search_router(
        {
            "enabled": True,
            "reranker_model_endpoint": "https://reranker.local",
            "reranker_model": "reranker-model",
            "reranker_enabled": True,
            "reranker_timeout_ms": 333,
            "reranker_max_context_chars": 555,
            "reranker_max_candidates": 7,
            "reranker_max_output_tokens": 777,
        },
        query_refiner_headers={"CF-Access-Client-Id": "id"},
    )

    assert router.reranker is not None
    assert router.reranker.config.model_endpoint == "https://reranker.local"
    assert router.reranker.config.model == "reranker-model"
    assert router.reranker.config.timeout_ms == 333
    assert router.reranker.config.max_context_chars == 555
    assert router.reranker.config.max_candidates == 7
    assert router.reranker.config.max_output_tokens == 777
    assert router.reranker.config.headers == {"CF-Access-Client-Id": "id"}


def test_build_search_router_does_not_alias_reranker_headers():
    headers = {"CF-Access-Client-Id": "id"}
    router = build_search_router(
        {"enabled": True, "reranker_model_endpoint": "https://reranker.local"},
        query_refiner_headers=headers,
    )

    headers["CF-Access-Client-Id"] = "changed"

    assert router.reranker is not None
    assert router.reranker.config.headers == {"CF-Access-Client-Id": "id"}
