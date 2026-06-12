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


def test_build_search_router_accepts_legacy_query_refiner_config_keys():
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

    assert router.query_refiner is not None
    assert router.query_refiner.config.model_endpoint == "https://legacy-query-refiner.local"
    assert router.query_refiner.config.model == "legacy-model"
    assert router.query_refiner.config.timeout_ms == 123
    assert router.query_refiner.config.max_context_chars == 456
    assert router.query_refiner.config.max_output_tokens == 789
    assert router.query_refiner.config.max_queries == 2


def test_build_search_router_new_query_refiner_config_keys_override_legacy():
    router = build_search_router(
        {
            "enabled": True,
            "model_endpoint": "https://legacy.local",
            "query_refiner_model_endpoint": "https://new.local",
            "model": "legacy-model",
            "query_refiner_model": "new-model",
            "planner_enabled": False,
            "query_refiner_enabled": True,
            "planner_timeout_ms": 111,
            "query_refiner_timeout_ms": 222,
            "planner_max_context_chars": 333,
            "query_refiner_max_context_chars": 444,
            "planner_max_output_tokens": 555,
            "query_refiner_max_output_tokens": 666,
            "planner_max_queries": 1,
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
