"""Search module assembly."""

from __future__ import annotations

from typing import Any

from .providers import (
    DuckDuckGoHtmlProvider,
    MarginaliaHtmlProvider,
    MojeekHtmlProvider,
    SearxngHtmlProvider,
    WikipediaOpenSearchProvider,
)
from .planner import SearchPlanner, SearchPlannerConfig
from .search_router import SearchConfig, SearchProviderConfig, SearchRouter


def build_search_router(
    search_config: dict[str, Any] | None = None,
    *,
    planner_headers: dict[str, str] | None = None,
) -> SearchRouter:
    """Build a config-driven search router with explicit provider allowlist."""
    search_config = search_config or {}
    planner_headers = planner_headers or {}
    provider_settings = search_config.get("providers", {})
    providers = {
        "duckduckgo_html": DuckDuckGoHtmlProvider(),
        "marginalia_html": MarginaliaHtmlProvider(),
        "mojeek_html": MojeekHtmlProvider(),
        "wikipedia_opensearch": WikipediaOpenSearchProvider(),
    }

    searxng_endpoint = (
        provider_settings.get("searxng_html", {}) or {}
    ).get("endpoint")
    if searxng_endpoint:
        providers["searxng_html"] = SearxngHtmlProvider(searxng_endpoint)

    provider_configs = {
        provider_id: _build_provider_config(provider_id, provider_settings)
        for provider_id in providers
    }

    config = SearchConfig(
        enabled=bool(search_config.get("enabled", False)),
        default_provider=str(search_config.get("default_provider", "duckduckgo_html")),
        timeout_ms=int(search_config.get("timeout_ms", 7000)),
        max_results=int(search_config.get("max_results", 10)),
        max_total_results=int(
            search_config.get(
                "search_max_total_results",
                search_config.get("max_results", 10),
            )
        ),
        cache_ttl_seconds=int(search_config.get("cache_ttl_seconds", 900)),
        max_body_bytes=int(search_config.get("max_body_bytes", 2_500_000)),
        max_retries=int(search_config.get("max_retries", 1)),
    )
    planner_config = SearchPlannerConfig(
        enabled=bool(
            search_config.get(
                "planner_enabled",
                bool(search_config.get("model_endpoint")),
            )
        ),
        model_endpoint=search_config.get("model_endpoint"),
        model=str(search_config.get("model", "search-planner")),
        timeout_ms=int(
            search_config.get(
                "planner_timeout_ms",
                search_config.get("timeout_ms", 7000),
            )
        ),
        max_context_chars=int(search_config.get("planner_max_context_chars", 12000)),
        max_output_tokens=int(search_config.get("planner_max_output_tokens", 512)),
        max_queries=int(search_config.get("planner_max_queries", 1)),
        headers=dict(planner_headers),
    )
    planner = (
        SearchPlanner(planner_config)
        if planner_config.enabled and planner_config.model_endpoint
        else None
    )

    return SearchRouter(
        config,
        providers=providers,
        provider_configs=provider_configs,
        planner=planner,
    )


def _build_provider_config(
    provider_id: str, provider_settings: dict[str, Any]
) -> SearchProviderConfig:
    defaults = {
        "duckduckgo_html": SearchProviderConfig(enabled=True, priority=10),
        "marginalia_html": SearchProviderConfig(enabled=True, priority=20),
        "mojeek_html": SearchProviderConfig(enabled=False, priority=30),
        "wikipedia_opensearch": SearchProviderConfig(
            enabled=False, priority=40, fallback_only=True
        ),
        "searxng_html": SearchProviderConfig(
            enabled=False, priority=50, fallback_only=True
        ),
    }

    config = defaults[provider_id]
    overrides = provider_settings.get(provider_id, {})
    if not overrides:
        return config

    return SearchProviderConfig(
        enabled=bool(overrides.get("enabled", config.enabled)),
        priority=int(overrides.get("priority", config.priority)),
        fallback_only=bool(overrides.get("fallback_only", config.fallback_only)),
        cache_ttl_seconds=(
            int(overrides["cache_ttl_seconds"])
            if "cache_ttl_seconds" in overrides
            else config.cache_ttl_seconds
        ),
        min_interval_seconds=float(
            overrides.get("min_interval_seconds", config.min_interval_seconds)
        ),
    )
