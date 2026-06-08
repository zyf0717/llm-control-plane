from src.search import build_search_router


def test_build_search_router_injects_planner_headers():
    router = build_search_router(
        {
            "enabled": True,
            "model_endpoint": "https://planner.local",
            "providers": {"duckduckgo_html": {"enabled": True}},
        },
        planner_headers={"CF-Access-Client-Id": "id"},
    )

    assert router.planner is not None
    assert router.planner.config.headers == {"CF-Access-Client-Id": "id"}


def test_build_search_router_does_not_alias_planner_headers():
    headers = {"CF-Access-Client-Id": "id"}
    router = build_search_router(
        {"enabled": True, "model_endpoint": "https://planner.local"},
        planner_headers=headers,
    )

    headers["CF-Access-Client-Id"] = "changed"

    assert router.planner is not None
    assert router.planner.config.headers == {"CF-Access-Client-Id": "id"}
